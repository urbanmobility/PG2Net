# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division


import torch
import numpy
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
from collections import deque, Counter

class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, cid_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                epoch_max=30, data_path='../data/', save_path='../results/', data_name='foursquare'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(open(self.data_path + self.data_name + '.pk', 'rb'),encoding='iso-8859-1')
        self.vid_list = data['vid_list']
        self.cid_list = data['cid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']
        self.tim_size = 48
        self.loc_size = len(self.vid_list)
        self.cid_size = len(self.cid_list)
        self.uid_size = len(self.uid_list)

        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.cid_emb_size = cid_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        test_id = data_neural[u]["test"]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0],s[1],s[2]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0],s[1],s[2]) for s in sessions[train_id[j]]])

            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]

            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            history_cid = np.reshape(np.array([s[2] for s in history]), (len(history), 1))

            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_cid'] = Variable(torch.LongTensor(history_cid))
            trace['history_count'] = history_count

            loc_tim = history
            loc_tim.extend([(s[0],s[1],s[2]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            cid_np = np.reshape(np.array([s[2] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['cid'] = Variable(torch.LongTensor(cid_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace
        train_idx[u] = train_id

    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())
    train_queue = deque()  #引入一个队列
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])

        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue

def get_acc(target, scores):
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return acc, ndcg

class auxiliaryLoss(torch.nn.Module):
    def __init__(self):
        super(auxiliaryLoss, self).__init__()
    def forward(self, output1, output2):
        size = output1.shape[0]
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive/size

def run_simple(pred,data, run_idx,auxiliary_rate,mode, lr, clip, model, optimizer, criterion,time_sim_matrix = None,poi_distance_matrix = None,poi_cid_tim = None):
    run_queue = None
    auxiliary_c = auxiliaryLoss().cuda()
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)
    time_sim_matrix = time_sim_matrix
    poi_distance_matrix = poi_distance_matrix
    poi_cid_tim = poi_cid_tim
    users_acc = {}
    for c in range(queue_len):
        optimizer.zero_grad()
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0,0,0,0,0,0,0]
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        cid = data[u][i]['cid'].cuda()
        target = data[u][i]['target'].cuda()
        uid = Variable(torch.LongTensor([u])).cuda()

        target_len = target.data.size()[0]
        scores,hidden_state,target_emb = model(loc, tim,cid,uid,target,target_len,time_sim_matrix,poi_distance_matrix,poi_cid_tim)
        loss = criterion(scores, target)

        pre = torch.max(scores,1).indices
        pred.append([numpy.array(loc.squeeze().cpu()),numpy.array(pre.cpu())])

        auxiliary_loss = auxiliary_c(hidden_state,target_emb,0)

        loss_all = loss + auxiliary_rate * auxiliary_loss
        if mode == 'train':
            loss_all.backward()
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()

        elif mode == 'test':
            users_acc[u][0] += len(target)
            acc, ndcg = get_acc(target, scores) #[3,1]
            users_acc[u][1] += acc[2]  #Rec@1
            users_acc[u][2] += acc[1]  # Rec@5
            users_acc[u][3] += acc[0]  # Rec@10
            users_acc[u][4] += ndcg[2]  # NDCG@1
            users_acc[u][5] += ndcg[1]  # NDCG@5
            users_acc[u][6] += ndcg[0]  # NDCG@10
        total_loss.append(loss.data.cpu().numpy())
    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss,pred
    elif mode == 'test':
        users_rnn_acc = {}
        for u in users_acc:
            each_user_list = [0.0,0.0,0.0,0.0,0.0,0.0]
            each_user_list[0] = users_acc[u][1] / users_acc[u][0]
            each_user_list[1] = users_acc[u][2] / users_acc[u][0]
            each_user_list[2] = users_acc[u][3] / users_acc[u][0]
            each_user_list[3] = users_acc[u][4] / users_acc[u][0]
            each_user_list[4] = users_acc[u][5] / users_acc[u][0]
            each_user_list[5] = users_acc[u][6] / users_acc[u][0]
            users_rnn_acc[u] = each_user_list
        avg_acc = [0.0,0.0,0.0,0.0,0.0,0.0]
        for i in range(6):
            avg_acc[i] = np.mean([users_rnn_acc[x][i] for x in users_rnn_acc])
        return avg_loss, avg_acc, users_rnn_acc,pred

