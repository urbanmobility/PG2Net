# coding: utf-8
"""PLSPL Model"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, long_emb,uids_emb):#(11,420) (1,420)
        uid_size = uids_emb.size()[0]
        seq_len = long_emb.size()[0]  # 11
        attn_energies = Variable(torch.zeros(uid_size,seq_len)).cuda()  # [1,11]
        for i in range(uid_size):
            for j in range(seq_len):
                attn_energies[i,j] = self.score(uids_emb[i], long_emb[j])
        return F.softmax(attn_energies)  # [1,11]

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy

class Model(nn.Module):
    def __init__(self,parameters, emb_size=500, hidden_units=500, dropout=0.8, user_dropout=0.5, data_neural = None):
        super(self.__class__, self).__init__()
        self.n_users = parameters.uid_size #937
        self.n_cids = parameters.cid_size #376
        self.n_items = parameters.loc_size#14001
        self.hidden_units = parameters.hidden_size #500
        self.loc_emb_size = parameters.loc_emb_size
        self.cid_emb_size = parameters.cid_emb_size
        self.uid_emb_size = parameters.uid_emb_size
        self.tim_emb_size = parameters.tim_emb_size

        self.long_emb = self.loc_emb_size + self.cid_emb_size + self.tim_emb_size #420
        self.short_loc_emb = self.uid_emb_size + self.loc_emb_size + self.tim_emb_size #370
        self.short_cid_emb = self.uid_emb_size + self.cid_emb_size + self.tim_emb_size #170
        self.loc_emb = nn.Embedding(self.n_items, self.loc_emb_size) #(14001,300)
        self.tim_emb = nn.Embedding(48, self.tim_emb_size) #(48,20)
        self.uid_emb = nn.Embedding(self.n_users, self.uid_emb_size)#(937,50)
        self.cid_emb = nn.Embedding(self.n_cids, self.cid_emb_size)#(376,100)
        self.attn = Attn("dot", 1)  # (dot,1)
        self.long_linear = nn.Linear(self.uid_emb_size,self.long_emb) #(50,420)
        self.long_pre = nn.Linear(self.long_emb,self.n_items )  # (420,14001)
        self.short_loc_linear = nn.Linear(hidden_units,self.n_items ) # (500,14001)
        self.short_cid_linear = nn.Linear(hidden_units,self.n_items ) # (500,14001)

        self.lstmcell_loc = nn.LSTM(input_size=self.short_loc_emb, hidden_size=hidden_units) #(370,500)
        self.lstmcell_cid = nn.LSTM(input_size=self.short_cid_emb, hidden_size=hidden_units) #(170,500)
        self.linear = nn.Linear(hidden_units*2 , self.n_items) #(1000,14001)
        self.dropout = nn.Dropout(0.3)
        self.data_neural = data_neural
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)
    """
    user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch,
    sequence_tim_batch, True, poi_distance_matrix, sequence_dilated_rnn_index_batch
    """
    #loc,tim,cid,uid,target,target_len
    def forward(self,loc,tim,cid,uid,target,target_len):#[21, 1],[21, 1],[21, 1],[10]
        h1 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()  # (1,32,500)
        c1 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()  # (1,32,500)
        h2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()  # (1,32,500)
        c2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()  # (1,32,500)

        history_items = self.loc_emb(loc[:-target_len]) #(11,1,300)
        history_tims = self.tim_emb(tim[:-target_len])  # 11,1,20)
        history_cids = self.cid_emb(cid[:-target_len])  # (11,1,100)

        uids = self.uid_emb(uid).unsqueeze(0)  # (1,1,50)
        #uids_long = uids.repeat(history_items.shape[0],1,1)# (11,1,50)
        uids_short = uids.repeat(target_len,1, 1)  # (10,1,50)

        current_items = self.loc_emb(loc[-target_len:])# (10,1,300)
        current_tims = self.tim_emb(tim[-target_len:])# (10,1,20)
        current_cids = self.cid_emb(cid[-target_len:])# (10,1,100)

        #long_term
        long_emb = torch.cat((history_items,history_tims,history_cids), 2) #(11,1,420)
        uids_emb = self.long_linear(uids)  # (1,1,420)
        #short_term
        short_loc_emb = torch.cat((uids_short,current_tims,current_items), 2)#(10,1,370)
        short_cid_emb = torch.cat((uids_short,current_tims,current_cids), 2)  #(10,1,170)
        # long_term
        long_attn = self.attn(long_emb.squeeze(1),uids_emb.squeeze(0)) #(11,420) + (1,420) -- >[1, 11]
        long_attn_res = long_attn.mm(long_emb.squeeze(1)) #(1,11) * (11,420) --> (1,420)
        long_attn_res = long_attn_res.repeat(target_len,1)
        long_predict = self.long_pre(long_attn_res)
        long_predict = F.selu(self.dropout(long_predict))#[10,14001]
        # short_term
        hidden_current_loc, (h1, c1) = self.lstmcell_loc(short_loc_emb, (h1, c1)) #(10,1,500)
        hidden_current_cid, (h2, c2) = self.lstmcell_cid(short_cid_emb, (h2, c2)) #(10,1,500)
        short_loc_predict = self.short_loc_linear(hidden_current_loc.squeeze(1))
        short_cid_predict = self.short_cid_linear(hidden_current_cid.squeeze(1))
        short_loc_predict = F.selu(self.dropout(short_loc_predict))
        short_cid_predict = F.selu(self.dropout(short_cid_predict))
        long_score = F.log_softmax(long_predict)
        short_loc_score = F.log_softmax(short_loc_predict)
        short_cid_score = F.log_softmax(short_cid_predict)
        score = F.log_softmax(long_score + short_loc_score + short_cid_score) #[10, 14001]
        return score