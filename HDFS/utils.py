#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np

import csv
import pickle
#import tensorflow as tf
import json


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)        

class Data():
    def __init__(self, data, shuffle=True):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices    
    
    def get_Embedding(self, inputs, targets, row, line):
        
        hidden = np.zeros((row, line, 300))
        results = np.zeros((2, 300))
        finall = np.zeros((row, 300))
        with open("/home/ustc-1/ljm/log/logdeep-master/data/hdfs/event2semantic_vec.json", 'r', encoding='UTF-8') as f:
           d = json.load(f)
        d['29'] = d['0']
        d['30'] = d['1'] 
        d['0'] = [0]*300
        d['1'] = [0]*300
        for i in range(0, row):
            for j in range(0, line):
                hidden[i][j] = d[str(inputs[i][j].item())]
                #print(inputs[i][j])
                #print(d[str(inputs[i][j].item())])
        for i in range(0, 2):
            results[i] = d[str(i + 29)]
            #print(results[i])
        for i in range(0, row):
            finall[i] = d[str(targets[i].item() + 29)]
            #print(targets[i])
            #print(d[str(targets[i].item() + 29)])
        return hidden, finall, results

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            #print("inputs", u_input)
            node = np.unique(u_input)
            #print("node",node)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            u_t = np.zeros((max_n_node, max_n_node))
            count = np.zeros(len(node))
            for i in np.arange(len(u_input)):
                if u_input[i] == 0:
                    break
                for j in np.arange(i + 1, len(u_input) - 1):
                    if u_input[j] == 0:
                        break
                    u1 = np.where(node == u_input[i])[0][0]
                    u2 = np.where(node == u_input[j])[0][0]
                    if u1 == u2:
                        u_t[u1][u2] = u_t[u1][u2] + 1
                    else:
                        u_t[u1][u2] = u_t[u1][u2] + 1
                        u_t[u2][u1] = u_t[u2][u1] + 1
            #print(count)    
            #total = np.sum(count)
            #print("u_t",u_t)
            for i in np.arange(1,len(node)):
                total = np.sum(u_t,axis=0)
                #print("total",total)
                for j in np.arange(1, len(node)):
                    u_A[i][j] = u_t[i][j]/total[i]
            #print("u_A",u_A)
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return inputs, alias_inputs, A, items, mask, targets, max_n_node
