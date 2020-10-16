# -*- coding: utf-8 -*-
"""
@author: a-kojima
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)
    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

def moving_sum(x, back, forward):
    x_pad = F.pad(x, (back,forward, 0, 0), "constant", 0.0)
    kernel = torch.ones(1, 1, back + forward + 1, dtype=x.dtype).to(DEVICE)
    return F.conv1d(x_pad.unsqueeze(1), kernel).squeeze(1).to(DEVICE)


# ==================================================
# CTC decoder
# ==================================================
class CTCModel(nn.Module):
    def __init__(self, NUM_CLASSES):        
        super(CTCModel, self).__init__()
        self.ctc_output = nn.Linear(NUM_HIDDEN_NODES * 2, NUM_CLASSES + 1)

    def forward(self, h):
        h, lengths = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        prediction = self.ctc_output(h)
        return prediction

# ==================================================
# mocha decoder
# ==================================================
class Monotonic_attention_train(nn.Module):
    def __init__(self,
                 window):
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
                        
    def _lstmcell(self, x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        c = (torch.sigmoid(forgetgate) * c) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        s = torch.sigmoid(outgate) * torch.tanh(c)
        return s, c        

    def get_monotonic(self, p_, a_prev):                
        # skip last element
        f_d = 1.0 - p_[:, :-1]
        cum_prod_p = torch.exp(torch.cumsum(torch.log(torch.clamp(f_d, min=min_val, max=1)), dim=1)).to(DEVICE)
        cum_prod_p = torch.cat((torch.ones(BATCH_SIZE, 1, requires_grad=True).to(DEVICE), cum_prod_p), dim=1)        
        div_ = a_prev / torch.clamp(cum_prod_p, min_val, 1.0)
        cum_sum = torch.cumsum(div_, dim=1)
        q = cum_prod_p * cum_sum
        a_ = p_ * q
        a_ = torch.clamp(a_, min=min_val, max=1.0)        
        return a_
    
    def get_chunk_wise(self, monotonic_attention, e_chunk):
        batch_size, max_seq = e_chunk.size()
        e_chunk -= torch.max(e_chunk, dim=1, keepdim=True)[0]
        exp_e_chunk = torch.exp(e_chunk)
        exp_e_chunk = torch.clamp(exp_e_chunk, min=1e-5)
        beta = exp_e_chunk * moving_sum(monotonic_attention / torch.clamp(moving_sum(exp_e_chunk, self.window-1, 0), 1e-10, float('inf')), 0, self.window-1)
        return beta
    
    def get_context_vector(self, chunk_wise_attention, enc_output_, length_enc_):    
        batch_size, max_seq, _ = enc_output_.size()
        context_vec = enc_output_ * ((chunk_wise_attention).unsqueeze(2))
        return torch.sum(context_vec, dim=1)
    
    def forward(self, enc_output_, target, length_enc_):        
        enc_output_, lengths = nn.utils.rnn.pad_packed_sequence(enc_output_, batch_first=True)
        # =========================================
        # batch inro analysis
        # =========================================
        batch_size, max_seq, _ = enc_output_.size()
        number_of_step = target.size()[1]
        
        # =========================================
        # init
        # =========================================        
        prediction = torch.zeros((batch_size, number_of_step, NUM_CLASSES), requires_grad=False).to(DEVICE)                
        # mask for calculating context vector
        mask = torch.zeros((BATCH_SIZE, max_seq), requires_grad=False).to(DEVICE)      
        for po in range(0, len(length_enc_)):
            mask.data[po, int(length_enc_[po]):] = 1.0                
        mask = mask.data.bool()
        
        # init previous monotonic attention
        a_prev = torch.zeros((batch_size, max_seq), requires_grad=False).to(DEVICE)
        # special case
        a_prev.data[:, 0] = 1.0 
                
        # decoder state
        s = torch.zeros((batch_size, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c = torch.zeros((batch_size, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)

        # =========================================
        # prediction step
        # =========================================                      
        for step_ in range(0, number_of_step):
            
            # ================================================
            # calculate monotonic attention
            # ================================================        
            # B * T * F
            tmp1 = torch.tanh(self.W_h_mono(enc_output_.to(DEVICE)) + self.W_s_mono(s).unsqueeze(1))            
            #  F * B            
            v_norm_mono = self.g_mono / torch.norm(self.v_mono.weight, p=2)            
            # B * F
            g_v_mono = v_norm_mono.unsqueeze(1) * tmp1            
            e_mono = self.v_mono(g_v_mono) + self.r_mono
            e_mono = e_mono[:, :, 0]
            e_mono.masked_fill_(mask, -float('inf'))
            p_mono = torch.sigmoid((e_mono + torch.normal(mean=torch.zeros(e_mono.size()), std=1).to(DEVICE))).to(DEVICE)
            monotonic_attention = self.get_monotonic(p_mono, a_prev).to(DEVICE)            
            
            # ================================================
            # calculate chunkwise attention
            # ================================================    
            tmp2 = torch.tanh(self.W_h_chunk(enc_output_) + self.W_s_chunk(s).unsqueeze(1))
            e_chunk = self.v_chunk(tmp2)
            e_chunk = e_chunk[:, :, 0]            
            e_chunk.masked_fill_(mask, -float('inf'))
            chunk_wise_attention = self.get_chunk_wise(monotonic_attention, e_chunk).to(DEVICE)
            
            # ================================================
            # calculate context vector
            # ================================================            
            context_vector = self.get_context_vector(chunk_wise_attention, enc_output_, length_enc_).to(DEVICE)

            # ================================================
            # predicition
            # ================================================
            # B * CLASS
            rec_input = self.L_ys(target[:, step_]) + self.L_ss(s) + self.L_gs(context_vector)
            s, c = self._lstmcell(rec_input, c)
            y = self.L_yy(torch.tanh(self.L_gy(context_vector) + self.L_sy(s)))                        
            prediction[:, step_] = y
            
            # save monotonic attention for next prediction
            a_prev = monotonic_attention.detach().clone()
             
        return prediction

# ==================================================
# encoder
# ==================================================
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=FEATURE_DIM, hidden_size=NUM_HIDDEN_NODES, num_layers=NUM_ENC_LAYERS, batch_first=True, dropout=DROP_OUT, bidirectional=True)
    def forward(self, x, lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        h, (hy, cy) = self.bi_lstm(x)
        return h

# ==================================================
# define model
# ==================================================
class Model(nn.Module):
    def __init__(self, window=2):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.mocha_decoder = Monotonic_attention_train(window)
        self.ctc_decoder = CTCModel(NUM_CLASSES)
    
    def forward(self, speech, lengths_enc, target):
        h = self.encoder(speech, lengths_enc)
        mocha_prediction = self.mocha_decoder(h, target, lengths_enc)
        ctc_prediction = self.ctc_decoder(h)
        return mocha_prediction, ctc_prediction


if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================================================================
    # ASR model details
    # ==================================================================
    FEATURE_DIM = 40
    NUM_HIDDEN_NODES = 60 
    NUM_ENC_LAYERS = 4
    NUM_CLASSES = 3000
    WINDOW = 4 # mocha window
    min_val = 10 ** (-10)
        
    # ==================================================================
    # training stategy
    # ==================================================================        
    DROP_OUT = 0.2
            
    # ==================================================================
    # define model  and optimizer
    # ==================================================================    
    model = Model().to(DEVICE)
    model.apply(init_weight)
    model.train()
    currlr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = currlr, weight_decay=1e-5)    
    
    # ==================================================================
    # make toy data
    # ==================================================================    
    BATCH_SIZE = 4
    num_of_step = 20
    len_x = torch.tensor([300, 300, 300, 200])
    len_y = torch.tensor([3, 10, 15, 20]).int()
    x = torch.randn(BATCH_SIZE, torch.max(len_x), FEATURE_DIM)
    y = torch.LongTensor(BATCH_SIZE, num_of_step).random_(1, NUM_CLASSES)
    y[:, 0] = 0 # <sos>
    
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    len_y = len_y.to(DEVICE)
    
    # ==================================================================
    # feed data (forward)
    # ==================================================================    
    mocha_predict, ctc_predict = model(x, len_x, y)

    # ==================================================================
    # calculate loss
    # ==================================================================        
    cross_entropy_loss = 0.0
    # cross entropy
    for i in range(BATCH_SIZE):
        num_labels = len_y[i]
        label = y[i, :num_labels]
        onehot_target = torch.zeros((len(label), NUM_CLASSES), dtype=torch.float32, device=DEVICE)
        for j in range(len(label)):
            onehot_target[j][label[j]] = 1.0
        ls_target = 0.9 * onehot_target + ((1.0 - 0.9) / (NUM_CLASSES - 1)) * (1.0 - onehot_target)
        cross_entropy_loss += -(F.log_softmax(mocha_predict[i][:num_labels], dim=1) * ls_target).sum()            
    cross_entropy_loss = cross_entropy_loss / BATCH_SIZE
    
    # ctc loss
    ctc_loss = F.ctc_loss(F.log_softmax(ctc_predict, dim=2).transpose(0,1),
                         y,
                         len_x.tolist(),
                         (np.array(len_y.tolist())).tolist(),
                         blank = NUM_CLASSES)

    print('mocha loss:', cross_entropy_loss)
    print('ctc loss:', ctc_loss)
    loss = cross_entropy_loss + ctc_loss
    print('loss=', loss)
    
    # ==================================================================
    # backward
    # ==================================================================            
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


