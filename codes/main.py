# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import json
import time
import argparse
import numpy as np
from json import encoder
import gensim
from utils import caculate_time_sim
from train import run_simple, RnnParameterData,  markov,generate_input_long_history
from model import PG2Net
from collections import defaultdict

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
#导入location的预训练模型
wvmodel = gensim.models.KeyedVectors.load_word2vec_format("loc.emb",binary=False,encoding='utf-8')
vocab_size=len(wvmodel.vocab)+1
vector_size=wvmodel.vector_size #500
weight = torch.randn(vocab_size, vector_size)
words= wvmodel.wv.vocab #(loc,vec)
word_to_idx = {word: int(word) for _, word in enumerate(words)} #loc:index
word_to_idx['<unk>'] = 0
#print(word_to_idx) '78':78
idx_to_word = {int(word): word for _, word in enumerate(words)} #index:(loc,vec)
idx_to_word[0] = '<unk>'
for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    vector=wvmodel.wv.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]])
    weight[index, :] = torch.from_numpy(vector)

#导入location category的预训练模型
wvmodel = gensim.models.KeyedVectors.load_word2vec_format("cid.emb",binary=False,encoding='utf-8')
vocab_size=len(wvmodel.vocab)+1
vector_size=wvmodel.vector_size #50
weight_cid = torch.randn(vocab_size, vector_size)
words= wvmodel.wv.vocab #(loc,vec)

word_to_idx = {word: int(word) for _, word in enumerate(words)}
word_to_idx['<unk>'] = 0

idx_to_word = {int(word): word for _, word in enumerate(words)}
idx_to_word[0] = '<unk>'
for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    vector=wvmodel.wv.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]])
    weight_cid[index, :] = torch.from_numpy(vector)

def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size, #500，40
                                  cid_emb_size=args.cid_emb_size, tim_emb_size=args.tim_emb_size, #50，10
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p, #500，0.3
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2,
                                  optim=args.optim,clip=args.clip, epoch_max=args.epoch_max,
                                  data_path=args.data_path, save_path=args.save_path)

    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'cid_emb_size': args.cid_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'clip': args.clip, 'epoch_max': args.epoch_max}


    auxiliary_rate = 0.05
    model = PG2Net(parameters=parameters,weight=weight,weight_cid=weight_cid).cuda()
    if args.pretrain == 1:
        model.load_state_dict(torch.load("../pretrain/" + args.model_mode + "/res.m"))

    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=parameters.lr,weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,factor=parameters.lr_decay, threshold=1e-3)
    lr = parameters.lr
    #衡量指标
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}
    candidate = parameters.data_neural.keys() #937个用户 0-937

    data_train, train_idx = generate_input_long_history(parameters.data_neural, 'train', candidate=candidate)
    data_test, test_idx = generate_input_long_history(parameters.data_neural, 'test', candidate=candidate)


    print('users:{} markov:{} train:{} test:{}'.format(len(candidate), avg_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]])))
    SAVE_PATH = args.save_path
    msg = 'users:{} markov:{} train:{} test:{}'.format(len(candidate), avg_acc_markov,
                                                       len([y for x in train_idx for y in train_idx[x]]),
                                                       len([y for x in test_idx for y in test_idx[x]]))
    with open(SAVE_PATH + "result.txt","a") as file:
        file.write(msg + "\n")
    file.close()
    tmp_path = 'checkpoint/'
    if not SAVE_PATH + tmp_path:
        os.mkdir(SAVE_PATH + tmp_path)

    #计算时间相似度,
    time_sim_matrix = caculate_time_sim(parameters.data_neural) #(48,48)
    #导入时间和cid类别关系
    poi_cid_tim = pickle.load(open('cid_time.pkl', 'rb'), encoding='iso-8859-1')
    #导入每个位置之间的空间距离
    poi_distance_matrix = pickle.load(open('distance.pkl', 'rb'), encoding='iso-8859-1')

    for epoch in range(parameters.epoch):
        pred = []
        st = time.time()
        if args.pretrain == 0:
            model, avg_loss,pred= run_simple(pred,data_train, train_idx,auxiliary_rate,'train', lr, parameters.clip, model, optimizer,
                                         criterion, parameters.model_mode,time_sim_matrix,poi_distance_matrix,poi_cid_tim)

            print('auxiliary_rate:{}'.format(auxiliary_rate))
            msg = 'auxiliary_rate:{}'.format(auxiliary_rate)
            with open(SAVE_PATH + "result.txt","a") as file:
                file.write(msg + "\n")
            file.close()
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            msg = '==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr)
            with open(SAVE_PATH + "result.txt","a") as file:
                file.write(msg + "\n")
            file.close()
            metrics['train_loss'].append(avg_loss)
        avg_loss, avg_acc, users_acc,pred = run_simple(pred,data_test, test_idx,auxiliary_rate,'test', lr, parameters.clip, model,
                                                  optimizer, criterion, parameters.model_mode,time_sim_matrix,poi_distance_matrix,poi_cid_tim)
        #print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))
        print('==>Rec@1:{:.4f} Rec@5:{:.4f} Rec@10:{:.4f} NDCG@1:{:.4f} NDCG@5:{:.4f} NDCG@10:{:.4f} Loss:{:.4f}'.format(avg_acc[0],avg_acc[1],avg_acc[2],avg_acc[3],avg_acc[4],avg_acc[5],avg_loss))
        msg = '==>Rec@1:{:.4f} Rec@5:{:.4f} Rec@10:{:.4f} NDCG@1:{:.4f} NDCG@5:{:.4f} NDCG@10:{:.4f} Loss:{:.4f}'.format(avg_acc[0],avg_acc[1],avg_acc[2],avg_acc[3],avg_acc[4],avg_acc[5],avg_loss)
        with open(SAVE_PATH + "result.txt","a") as file:
            file.write(msg + "\n")
        file.close()

        pickle.dump(pred, open("{}_our_nyc_loc.pkl".format(epoch), 'wb'))
        metrics['valid_loss'].append(avg_loss)
        metrics['accuracy'].append(avg_acc[0])
        metrics['valid_acc'][epoch] = users_acc
        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        scheduler.step(avg_acc[0])
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmax(metrics['accuracy'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            auxiliary_rate += 0.05
            print('load epoch={} model state'.format(load_epoch))

            msg = 'load epoch={} model state'.format(load_epoch)
            with open(SAVE_PATH + "result.txt","a") as file:
                file.write(msg + "\n")
            file.close()

        if epoch == 0:
            print('single epoch time cost:{}'.format(time.time() - st))
            msg = 'single epoch time cost:{}'.format(time.time() - st)
            with open(SAVE_PATH + "result.txt","a") as file:
                file.write(msg + "\n")
            file.close()
        if lr <= 0.9 * 1e-7:
            break
        if args.pretrain == 1:
            break

    mid = np.argmax(metrics['accuracy'])
    avg_acc = metrics['accuracy'][mid]
    load_name_tmp = 'ep_' + str(mid) + '.m'
    print("最优模型:",SAVE_PATH + tmp_path + load_name_tmp)
    return avg_acc

def load_pretrained_model(config):
    res = json.load(open("../pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.L2 = res["L2"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["cid_emb_size"]
        self.pretrain = 1


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--cid_emb_size', type=int, default=50, help="cid embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='foursquare_test_new')
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=40)
    parser.add_argument('--data_path', type=str, default='数据集路径')
    parser.add_argument('--save_path', type=str, default='文件保存路径')
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)
    ours_acc = run(args)