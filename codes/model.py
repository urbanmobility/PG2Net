# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attn_user(nn.Module):
    def __init__(self):
        super(Attn_user, self).__init__()
    def forward(self, user_emb, id_emb):  # (1,500) (11,500)
        seq_len = id_emb.size()[0]  # 11
        state_len = user_emb.size()[0]  # 1
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(user_emb[i], id_emb[j])
        return F.softmax(attn_energies)
    def score(self, hidden, encoder_output):
        energy = hidden.dot(encoder_output)
        return energy

class Attn_loc(nn.Module):
    def __init__(self):
        super(Attn_loc, self).__init__()
    def forward(self, history, current, poi_distance_matrix):
        seq_len = history.size()[0]  # 11
        state_len = current.size()[0]  # 10
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = 1/poi_distance_matrix[current[i], history[j]]
        return F.softmax(attn_energies)

class Attn_cid_time(nn.Module):
    def __init__(self):
        super(Attn_cid_time, self).__init__()
    def forward(self, history, current,cid_time):
        seq_len = history.size()[0]  # 11
        state_len = current.size()[0]  # 10
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = cid_time[current[i], history[j]]
        return F.softmax(attn_energies)

class Attn_time(nn.Module):
    def __init__(self):
        super(Attn_time, self).__init__()
    def forward(self, history, current, time_sim_matrix):
        seq_len = history.size()[0]  # 11
        state_len = current.size()[0]  # 10
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()  # [10, 11]
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = time_sim_matrix[current[i], history[j]]
        return F.softmax(attn_energies)  # [10, 11]

class Self_Attn_loc(nn.Module):
    def __init__(self):
        super(Self_Attn_loc, self).__init__()
    def forward(self, history, current, poi_distance_matrix):
        seq_len = history.size()[0]  # 10
        state_len = current.size()[0]  # 10
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(i + 1):
                attn_energies[i, j] = 1/poi_distance_matrix[current[i], history[j]]
        return F.softmax(attn_energies)

class Self_Attn_time(nn.Module):
    def __init__(self):
        super(Self_Attn_time, self).__init__()
    def forward(self, history, current, time_sim_matrix):
        seq_len = history.size()[0]  # 11
        state_len = current.size()[0]  # 10
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(i + 1):
                attn_energies[i, j] = time_sim_matrix[current[i], history[j]]
        return F.softmax(attn_energies)

class Self_Attn_cid_time(nn.Module):
    def __init__(self):
        super(Self_Attn_cid_time, self).__init__()
    def forward(self, history, current, cid_time):
        seq_len = history.size()[0]  # 10
        state_len = current.size()[0]  # 10
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(i + 1):
                attn_energies[i, j] = cid_time[current[i], history[j]]
        return F.softmax(attn_energies)

class PG2Net(nn.Module):
    def __init__(self, parameters, weight, weight_cid):
        super(PG2Net, self).__init__()
        self.loc_size = parameters.loc_size
        self.cid_size = parameters.cid_size
        self.uid_size = parameters.uid_size

        self.loc_emb_size = parameters.loc_emb_size
        self.cid_emb_size = parameters.cid_emb_size
        self.uid_emb_size = parameters.uid_emb_size

        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size

        self.hidden_size = parameters.hidden_size
        self.history_hidden_size = 250
        self.use_cuda = parameters.use_cuda  # True

        self.emb_loc = nn.Embedding.from_pretrained(weight, freeze=True)
        self.emb_cid = nn.Embedding.from_pretrained(weight_cid, freeze=True)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size + self.cid_emb_size  # 560

        self.attn_user = Attn_user()
        self.attn_cid_time = Attn_cid_time()
        self.attn_loc = Attn_loc()
        self.attn_time = Attn_time()
        self.self_attn_loc = Self_Attn_loc()
        self.self_attn_time = Self_Attn_time()
        self.self_attn_cid_time = Self_Attn_cid_time()
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        self.rnn_encoder = nn.LSTM(input_size, self.history_hidden_size, 1, bidirectional=True)
        self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(3 * self.hidden_size, self.loc_size)
        self.fc_final2 = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.fc_user = nn.Linear(self.uid_emb_size, self.hidden_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, cid, uid, target, target_len, time_sim_matrix, poi_distance_matrix, poi_cid_tim):
        h1 = Variable(torch.zeros(2, 1, self.history_hidden_size))  # (1,1,250)
        h2 = Variable(torch.zeros(1, 1, self.hidden_size))  # (1,1,500)
        c1 = Variable(torch.zeros(2, 1, self.history_hidden_size))  # (1,1,250)
        c2 = Variable(torch.zeros(1, 1, self.hidden_size))  # (1,1,500)
        if self.use_cuda:
            h1 = h1.cuda()
            h2 = h2.cuda()
            c1 = c1.cuda()
            c2 = c2.cuda()

        attn_cid_time = self.attn_cid_time(cid[:-target_len], tim[-target_len:], poi_cid_tim)
        attn_weights_loc = self.attn_loc(loc[:-target_len], loc[-target_len:], poi_distance_matrix)
        attn_weights_time = self.attn_time(tim[:-target_len], tim[-target_len:], time_sim_matrix)
        attn_weights_loc = attn_weights_loc.unsqueeze(0)
        attn_weights_time = attn_weights_time.unsqueeze(0)
        attn_cid_time = attn_cid_time.unsqueeze(0)

        self_attn_cid_time = self.self_attn_cid_time(cid[-target_len:], tim[-target_len:], poi_cid_tim)
        self_attn_weights_loc = self.self_attn_loc(loc[-target_len:], loc[-target_len:], poi_distance_matrix)
        self_attn_weights_time = self.self_attn_time(tim[-target_len:], tim[-target_len:], time_sim_matrix)
        self_attn_weights_loc = self_attn_weights_loc.unsqueeze(0)
        self_attn_weights_time = self_attn_weights_time.unsqueeze(0)
        self_attn_cid_time = self_attn_cid_time.unsqueeze(0)

        loc_emb = self.emb_loc(loc)
        target_emb = self.emb_loc(target)
        tim_emb = self.emb_tim(tim)
        cid_emb = self.emb_cid(cid)
        uid_emb = self.emb_uid(uid)

        x = torch.cat((loc_emb, tim_emb, cid_emb), 2)
        x = self.dropout(x)

        hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
        hidden_state, (h2, c2) = self.rnn_decoder(x[-target_len:], (h2, c2))

        hidden_history = hidden_history.squeeze(1)
        hidden_state = hidden_state.squeeze(1)
        target_emb = target_emb.squeeze(1)

        uid_emb = self.fc_user(uid_emb)
        user_attn_weights = self.attn_user(uid_emb, hidden_history).unsqueeze(0)
        context_user = user_attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)

        context_cidtime = attn_cid_time.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        context_loc = attn_weights_loc.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        context_time = attn_weights_time.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        context = context_loc + context_time + context_cidtime

        hidden_statet_cidtime = self_attn_cid_time.bmm(hidden_state.unsqueeze(0)).squeeze(0)
        hidden_state_loc = self_attn_weights_loc.bmm(hidden_state.unsqueeze(0)).squeeze(0)
        hidden_state_time = self_attn_weights_time.bmm(hidden_state.unsqueeze(0)).squeeze(0)
        hidden_state = hidden_state_loc + hidden_state_time + hidden_statet_cidtime

        context_user = context_user.repeat(hidden_state.shape[0], 1)
        hidden_state_all = hidden_state + uid_emb.repeat(hidden_state.shape[0], 1)

        out = torch.cat((hidden_state_all,context,context_user), 1)
        out = self.dropout(out)
        y = self.fc_final(out)
        y1 = self.fc_final2(out)
        score = F.log_softmax(y)
        return score, y1, target_emb
