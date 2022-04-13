#!/usr/bin/env python36
# -*- coding: utf-8 -*-


import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class MyGraph(Module):
    def __init__(self, opt, n_node):
        super(MyGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        self.mlp = self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 256), 
            nn.ReLU(),
            nn.Linear(256,128))

    def compute_scores(self, hidden, mask, finall, results):
        
        ht = finall
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q2)) #q1+q2
        #alpha = F.softmax(alpha, dim=1)
        #print(alpha)
        #print(hidden)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.mlp(a)
        return a, finall

    def forward(self, inputs, A):
        hidden = self.gnn(A, inputs)
        return hidden
    
    def metric_dot(self, a, b):
        return torch.matmul(a, b.transpose(1, 0))

    def metric_cosine(self, a, b):
        return torch.cosine_similarity(a, b, dim=1)

    def metric_euclid(self, a, b):
        return torch.norm(a - b, 2, 1)

    def metric_norm_euclid(self, a, b):
        v0 = v0/torch.norm(v0, 2, 1).view(-1,1)
        v1 = v1/torch.norm(v1, 2, 1).view(-1,1)
        return -torch.norm(v0-v1, 2, 1).view(-1,5)

    def metric_manhattan(self, a, b):
        return torch.sum(torch.abs(a - b), 1)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv) 



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    inputs, alias_inputs, A, items, mask, targets, max_n_node = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    shape = list(items.size())
    hidden, finall, results = data.get_Embedding(items, targets, shape[0], shape[1] )
    #print(hidden[1][1])
    hidden = trans_to_cuda(torch.Tensor(hidden).float())
    #print(hidden[1][1])
    results = trans_to_cuda(torch.Tensor(results).float())
    finall = trans_to_cuda(torch.Tensor(finall).float())
    hidden = model(hidden, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    a, finall = model.compute_scores(seq_hidden, mask, finall, results)
    return inputs, a, finall, results, targets


def train(model, train_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        _, a, finall, _ , targets= forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        #shape = list(finall.size())
        #y = torch.ones(shape[0], 1)
        #y = trans_to_cuda(torch.Tensor(y).int())
        loss = model.loss_function(a, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

def test(model, test_data):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit = []
    TP, FP, FN, TN = 0, 0, 0, 0
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        inputs, a, finall, results, targets = forward(model, i, test_data)
        #sub_scores = torch.cosine_similarity(a, finall, dim=1)
        #for x, target in zip(a, targets):
        # print(x)
        pred = a.data.max(1, keepdim=True)[1].view(-1)
        pred_res = np.array(pred.cpu())
        j = 0
        for pred, target in zip(pred_res, targets):
          #print(pred,   target)
          if ((pred == 0)&(target == 0)):
             TN +=1
          elif((pred == 1)&(target == 1)):
             TP +=1
          elif((pred == 0)&(target == 1)):
             FN +=1
          elif((pred == 1)&(target == 0)):
             FP +=1        

    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    #hit = np.mean(hit) * 100
    
    
    print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
            
    #print(hit)
#def final_Embedding():







