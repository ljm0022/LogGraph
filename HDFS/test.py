#!/usr/bin/env python36
# -*- coding: utf-8 -*-


import torch
import pickle
from utils import Data, split_validation
from model import *
import random


def main():
    test_data = pickle.load(open('test.txt', 'rb'))      
    test_data = Data(test_data, shuffle=True)
    model = torch.load('GNN_netyang.pkl')
    test(model, test_data)

if __name__ == '__main__':
    main()
