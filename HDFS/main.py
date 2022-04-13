#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
import torch
from utils import Data, split_validation
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=300, help='hidden state size')
parser.add_argument('--epoch', type=int, default=11, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('train.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)   

    model = trans_to_cuda(MyGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        train(model, train_data)

    end = time.time()
    
    torch.save(model, 'GNN_net.pkl')
    
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()



