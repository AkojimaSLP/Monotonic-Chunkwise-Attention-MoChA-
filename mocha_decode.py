# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:06:01 2020

@author: a-kojima
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as pl

def plot_attention(attention_list, title='mono') :
#    print('yuuu', len(attention_list), attention_list[0][0, :].size(), attention_list[0].size())
    att = torch.zeros(len(attention_list), attention_list[0][0, :].size()[0]).float()
    for i in range(0, len(attention_list)):
        att[i, :]=  attention_list[i][0, :]
    pl.figure()
    pl.imshow(att.detach().numpy(), aspect='auto')
    pl.title(title)


def frame_stacking(x):
    newlen = len(x) // 3
    stacked_x = x[0:newlen * 3].reshape(newlen, 40 * 3)
    del x
    return stacked_x
                
# ==================================================
# mocha
# ==================================================
class Monotonic_attention_train(nn.Module):
    def __init__(self,
                 window=4):
        super(Monotonic_attention_train, self).__init__()        
        self.window = window
        
        # ==================================
        # monotonic replated
        # ==================================
        self.W_s_mono = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES*2, bias=True)
        self.W_h_mono = nn.Linear(NUM_HIDDEN_NODES*2, NUM_HIDDEN_NODES*2, bias=False)
        self.v_mono = nn.Linear(NUM_HIDDEN_NODES*2, 1, bias=False)
        self.g_mono = torch.nn.Parameter(1.0 / torch.sqrt((torch.ones(1) * NUM_HIDDEN_NODES*2).float()).float(), requires_grad=True)
        self.r_mono = torch.nn.Parameter(torch.ones(1) * (- 4.0) , requires_grad=True)
        
        # ==================================
        # chunkwise related
        # ==================================
        self.W_s_chunk = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES*2, bias=True)
        self.W_h_chunk = nn.Linear(NUM_HIDDEN_NODES*2, NUM_HIDDEN_NODES*2, bias=False)
        self.v_chunk = nn.Linear(NUM_HIDDEN_NODES*2, 1, bias=False)          
        
        # ==================================
        # decoder related
        # ==================================
        self.L_sy = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, bias=False)
        self.L_gy = nn.Linear(NUM_HIDDEN_NODES*2, NUM_HIDDEN_NODES)
        self.L_yy = nn.Linear(NUM_HIDDEN_NODES, NUM_CLASSES)
        self.L_ys = nn.Embedding(NUM_CLASSES, NUM_HIDDEN_NODES * 4)
        self.L_ss = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES * 4, bias=False)
        self.L_gs = nn.Linear(NUM_HIDDEN_NODES*2, NUM_HIDDEN_NODES * 4)
        
        # ==================================
        # dropout  for beta
        # ==================================
        self.dropout = nn.Dropout(p=0.0)
                
    def _lstmcell(self, x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        c = (torch.sigmoid(forgetgate) * c) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        s = torch.sigmoid(outgate) * torch.tanh(c)
        return s, c                
        
    def get_chunkwise_decode(self, enc_out_, chunk_energy_, fired_index_):
        enc_output_for_chunk = torch.zeros(1, self.window, NUM_HIDDEN_NODES * 2)
        chunk_wise_win = torch.zeros(1, self.window)
        
        if fired_index_ < self.window:
            chunk_st = 0
            chunk_ed = fired_index_ + 1        
        else:            
            chunk_st = (fired_index_ + 1) - self.window
            chunk_ed = fired_index_ + 1                            
        # =================================
        # calc. chunkwise attention eficiently
        # =================================     
        enc_output_for_chunk[:, :(chunk_ed - chunk_st), :] = enc_out_[:, chunk_st:chunk_ed, :]
        chunk_wise_win[:, :(chunk_ed - chunk_st)] = chunk_energy_[:, chunk_st:chunk_ed]
        chunk_wise_win = F.softmax(chunk_wise_win, dim=1)        
        context_vector_ = enc_output_for_chunk * ((chunk_wise_win).unsqueeze(2))        
        return torch.sum(context_vector_, dim=1), chunk_wise_win
    
    
    def get_fired_frame(self, monotonic_filter_):
        # monotonic_filter_: B * length
        index_ = torch.where(monotonic_filter_[0, :]==1)[0]
        if len(index_) == 0:
            return torch.zeros(monotonic_filter_.size()), -1
        else:
            one_hot = torch.min(index_)
            aaa = torch.zeros(monotonic_filter_.size())
            aaa[:, one_hot] = 1
            return aaa, one_hot            
    
    def forward(self, enc_output_, x, encoder_):
        
        # =========================================
        # batch inro analysis
        # =========================================
        batch_size, max_seq, hidden_size_enc = enc_output_.size()
        
        # =========================================
        # init
        # =========================================        
        #  previous monotonic attention
        a_prev = torch.zeros((1, max_seq), requires_grad=False).to(DEVICE)        
        a_prev.data[:, 0] = 1.0                         
        # decoder state
        s = torch.zeros((1, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c = torch.zeros((1, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)        
        attention_list = []
        attention_list.append(a_prev)
        beta_list = []        
        is_decode = True
        count_same_position = 0
        hypes = [([], 0.0, c, s, enc_output_, attention_list, beta_list, count_same_position, is_decode)]
        
        finished_hypes = []        

        MAX_STEP = 80
        
        
        # =========================================
        # decoding loop
        # =========================================                             
        for step_ in range(0, MAX_STEP):
            
            new_hypes= []
            
            # ==========================================
            # 1 step decoding            
            # ==========================================            
            # get hypothesis
            for hype in hypes:
                # last decoder info
                out_seq, seq_score, c, s, encoder_state, attention_list, beta_list, count_same_, is_decode = hype                
                # get last attention (for  calculate hard monotonic)
                last_attention = attention_list[- 1]                               
                
                # ================================================
                # step1. calculate monotonic energy
                # ================================================        
                # B * T * F
                tmp1 = torch.tanh(self.W_h_mono(enc_output_.to(DEVICE)) + self.W_s_mono(s).unsqueeze(1))
                
                #  F * B            
                v_norm_mono = self.g_mono / torch.norm(self.v_mono.weight, p=2)
                
                # B * F
                g_v_mono = v_norm_mono.unsqueeze(1) * tmp1                    
    
                e_mono = self.v_mono(g_v_mono) + self.r_mono # 1 * max_seq * 1
                e_mono = e_mono[:, :, 0] 
                
                # sigmoid threshold
                p_mono = torch.sigmoid(e_mono).to(DEVICE)
                fired_frame_ = (p_mono >= SIGMOID_THRESHOLD).float()     

                # ==========================================                
                # calculate hard monotonic
                # ==========================================
                # ============================
                # last_attention: [0, 0, 0, 1, 0, 0, 0]
                #-> cumsum        [0, 0, 0, 1, 1, 1, 1]
                # fired_frame:    [1, 0, 1, 0, 1, 0, 1]
                # * ->            [0, 0, 0, 0, 1, 0, 1] 
                # ============================
                                
                p_select = fired_frame_ * torch.cumsum(last_attention, dim=1)                   
                attention, fired_index = self.get_fired_frame(p_select)
                
                # make sure same attention twice
                if torch.sum(attention * last_attention) == 1:
                    count_same_ = count_same_ + 1
                else:
                    count_same_ = 0
                    
                if count_same_ == 2:
                    attention = attention * 0
                
                # ===============================
                # attend or not
                # ===============================
                
                # attend
                if torch.sum(attention) != 0:                    
                    # ================================================
                    # calculate chunkwise attention
                    # ================================================        
                    tmp2 = torch.tanh(self.W_h_chunk(enc_output_) + self.W_s_chunk(s).unsqueeze(1))        
                    e_chunk = self.v_chunk(tmp2)
                    e_chunk = e_chunk[:, :, 0]                     
                    context_vector, beta = self.get_chunkwise_decode(enc_output_, e_chunk, fired_index)                                        
                    # B * CLASS
                    y = self.L_yy(torch.tanh(self.L_gy(context_vector) + self.L_sy(s)))    
                    scores = F.softmax(y, dim=1).data.squeeze(0)      
                    if step_ <= FIRST_BEAM_STEP:                        
                        best_scores, indices = scores.topk(FIRST_BEAM_SIZE)                        
                    else:
                        best_scores, indices = scores.topk(BEAM_SIZE)      
                        
                    new_attention_list = attention_list + [attention]    
                    new_beta_list = beta_list + [beta]                    
                    for score, index in zip(best_scores, indices):                                              
                        new_seq = out_seq + [index]
                        new_seq_score = seq_score + score
                        rec_input = self.L_ys(index) + self.L_ss(s) + self.L_gs(context_vector)
                        new_s, new_c = self._lstmcell(rec_input, c)
                        # sppear <eos>
                        if int(index) == 1:
                            # reset encoder
                            new_encoder_state = encoder_.get_new_hidden_state(x, fired_index)
                            new_s = new_s * 0
                            new_c = new_c * 0
                            new_flag = True
                        else:
                            new_encoder_state = encoder_state
                            new_flag = True
                        new_hypes.append((new_seq, new_seq_score, new_c, new_s, new_encoder_state, new_attention_list, new_beta_list, count_same_, new_flag ))                    
                # not attend
                else:       
                    new_attention_list = attention_list + [attention]
                    new_seq = out_seq + [1]                    
                    new_seq_score = seq_score
                    new_c = c
                    new_s = s
                    new_encoder_state = encoder_state
                    new_beta_list = beta_list
                    new_flag = False
                    new_hypes.append((new_seq, new_seq_score, new_c, new_s, new_encoder_state, new_attention_list, new_beta_list, count_same_, new_flag))
                                    
            # ==========================================
            # pruning
            # ==========================================            
            # step1 remove hypes 
            tmp_hypes = []
            for ll in range(0, len(new_hypes)):
                # <eos> appears or rearch maximum frames
                #if int(new_hypes[ll][0][-1]) == 1 and new_hypes[ll][-1] == False:
                if new_hypes[ll][-1] == False:
                    # limit hypothesis length                    
                    if len(new_hypes[ll][0]) >= 2 and new_hypes[ll] not in finished_hypes:
                        finished_hypes.append(new_hypes[ll])
                else:
                    if new_hypes[ll] not in tmp_hypes:
                        # continue...
                        tmp_hypes.append(new_hypes[ll])            
            hypes = tmp_hypes
            
            if len(hypes) == 0:
                break
            print('activate', len(hypes))
                                        
            # ===============================
            # pruning decoding hypothesis
            # ===============================                                   
            new_hypes_sorted = sorted(hypes, key=itemgetter(1), reverse=True)
            if step_ <= FIRST_BEAM_STEP:
                hypes = new_hypes_sorted[:FIRST_BEAM_SIZE]            
            else:
                hypes = new_hypes_sorted[:BEAM_SIZE]            
                            
        # ===============================
        # select best hypothesis
        # ===============================                        
        if len(finished_hypes) != 0:
            ffh = finished_hypes[0] # best result
            plot_attention(ffh[5], 'monotonic')            
        return ffh

class CTCModel(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(CTCModel, self).__init__()
        self.ctc_output = nn.Linear(NUM_HIDDEN_NODES * 2, NUM_CLASSES + 1)

    def forward(self, h):
        h, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        prediction = self.ctc_output(h)
        return prediction
    

# ==================================================
# encoder
# ==================================================
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.uni_lstm = nn.LSTM(input_size=LMFB_DIM*3, hidden_size=NUM_HIDDEN_NODES, num_layers=NUM_ENC_LAYERS, batch_first=True, dropout=DROP_OUT, bidirectional=IS_BLSTM)
        
    def forward(self, x):
        h, (hy, cy) = self.uni_lstm(x)
        return h
    
    def get_new_hidden_state(self, x, frame_index):
        original_frame = x.size()[1]        
        new_out_hidden_states = torch.zeros(1, original_frame, NUM_HIDDEN_NODES * 2)
        if frame_index >= original_frame:
            return []
        h, (_, _) = self.uni_lstm(x[:, frame_index:, :])
        new_out_hidden_states[:, frame_index:, :] = h
        return new_out_hidden_states

# ==================================================
# define model
# ==================================================
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.mocha_decoder = Monotonic_attention_train()
        self.ctc_decoder = CTCModel(NUM_CLASSES)
    
    def forward(self, speech):
        h = self.encoder(speech)
        mocha_prediction = self.mocha_decoder(h, speech, self.encoder)
#        ctc_prediction = self.ctc_decoder(h)
        return mocha_prediction

if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================================================================
    # ASR model
    # ==================================================================
    feature_format = 'npy'
    LMFB_DIM = 40
    LRDECAY = True
    NUM_HIDDEN_NODES = 512
    LAS_HIDDEN_NODE = 512
    NUM_ENC_LAYERS = 4
    NUM_DEC_LAYERS = 1
    NUM_ENC_LAYERS = 4
    NUM_CLASSES = 3672
    ENCODER_TYPE = 'lstm'
    WINDOW = 4
    IS_BLSTM = True
        
    # ==================================================================
    # training stategy
    # ==================================================================    
    BATCH_SIZE = 1 #20
    MAX_FRAME = 3000
    MIN_FRAME = 20
    MAX_LABEL = 1000
    DROP_OUT = 0
    min_val = 10 ** (-10)    
    

    # ==================================================================
    # decoding stategy
    # ==================================================================    
    WINDOW = 4 #4
    SIGMOID_THRESHOLD = 0.5 
    
    # beam
    BEAM_SIZE = 10  
    FIRST_BEAM_SIZE = 10
    FIRST_BEAM_STEP = 1000
    scaling_chunkwise = 1.0 #1.0
    
    x_file = '<npy_path>'
    model_path = '<model_path>'

    model = Model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))        
    model.eval()
    
    cpudat = np.load(x_file)
    cpudat = frame_stacking(cpudat)
    xs = torch.from_numpy(cpudat).to(DEVICE).float().unsqueeze(0)
    prediction_beam = model(xs)
