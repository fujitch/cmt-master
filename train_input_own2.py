# -*- coding: utf-8 -*-
"""
隠れ層2層のニューラルネット活性化関数relu→softmaxの学習済みモデル
で、目的の出力値に最も反応する入力を学習させる
"""

import pickle
import numpy as np
import chainer.functions as F
import chainer
import matplotlib.pyplot as plt


model = pickle.load(open('dnn_latest_64_1_5000Hz_drop001.pkl', 'rb'))

input_nodes = 100
hidden_nodes1 = 32
hidden_nodes2 = 8
output_nodes = 3
learning_rate = 0.1
epochs = 1000000
dropout_ratio = 0.01

w1 = model.l1.W.data
w2 = model.l2.W.data
w3 = model.l3.W.data

for k in [0, 1, 2]:
    output = np.zeros((output_nodes))
    output = np.array(output, dtype=np.float32)
    output = output[:, np.newaxis]
    
    output[k] = 1
    
    default_input = np.ones((hidden_nodes1))
    default_input = np.array(default_input, dtype=np.float32)
    default_input = default_input[:, np.newaxis]
    
    for epoch in range(epochs):
        y = np.dot(w2, default_input)
        del_y = np.ones((hidden_nodes2))
        del_y = del_y[:, np.newaxis]
        for i in range(hidden_nodes2):
            if y[i] < 0:
                y[i] = 0
                del_y[i] = 0
        y = np.dot(w3, y)
        y = chainer.Variable(y.T)
        y = F.softmax(y).data.T
        
        diff = y - output
        diff = np.dot(w3.T, diff)
        
        default_input -= learning_rate * np.dot(w2.T, diff * del_y)
        
        for i in range(hidden_nodes1):
            if default_input[i] < 0:
                default_input[i] = 0
    
        if epoch%10000 == 0:
            print('[pre1]')
            print(sum(abs(diff)))
        
    if k == 0:
        hidden_best1 = default_input
    elif k == 1:
        hidden_best2 = default_input
    elif k == 2:
        hidden_best3 = default_input
        
for k in [0, 1, 2]:
    if k == 0:
        output = hidden_best1
    elif k == 1:
        output = hidden_best2
    elif k == 2:
        output = hidden_best3
        
    default_input = np.ones((input_nodes))
    default_input = np.array(default_input, dtype=np.float32)
    default_input = default_input[:, np.newaxis]
    
    for epoch in range(10000):
        y = np.dot(w1, default_input)
        del_y = np.ones((hidden_nodes1))
        del_y = del_y[:, np.newaxis]
        for i in range(hidden_nodes1):
            if y[i] < 0:
                y[i] = 0
                del_y[i] = 0
        
        diff = y - output
        
        default_input -= learning_rate * np.dot(w1.T, diff * del_y)
        
        for i in range(input_nodes):
            if default_input[i] < 0:
                default_input[i] = 0
    
        if epoch%1000 == 0:
            print('[pre2]')
            print(sum(abs(diff)))
        
    if k == 0:
        dummy_best1 = default_input
    elif k == 1:
        dummy_best2 = default_input
    elif k == 2:
        dummy_best3 = default_input

for k in [0, 1, 2]:
    output = np.zeros((output_nodes))
    output = np.array(output, dtype=np.float32)
    output = output[:, np.newaxis]
    
    output[k] = 1
    
    if k == 0:
        default_input = dummy_best1
    elif k == 1:
        default_input = dummy_best2
    elif k == 2:
        default_input = dummy_best3
    
    w1 = model.l1.W.data
    w2 = model.l2.W.data
    w3 = model.l3.W.data
    
    for epoch in range(epochs):
        y = np.dot(w1, default_input)
        del_y1 = np.ones((hidden_nodes1))
        del_y1 = del_y1[:, np.newaxis]
        for i in range(hidden_nodes1):
            if y[i] < 0:
                y[i] = 0
                del_y1[i] = 0
        y = np.dot(w2, y)
        del_y2 = np.ones((hidden_nodes2))
        del_y2 = del_y2[:, np.newaxis]
        for i in range(hidden_nodes2):
            if y[i] < 0:
                y[i] = 0
                del_y2[i] = 0
        
        y = np.dot(w3, y)
        y = chainer.Variable(y.T)
        y = F.softmax(y).data.T
        
        diff = y - output
        diff = np.dot(w3.T, diff)
        diff = np.dot(w2.T, diff * del_y2)
        
        default_input -= learning_rate * np.dot(w1.T, diff * del_y1)
        
        for i in range(input_nodes):
            if default_input[i] < 0:
                default_input[i] = 0
                             
        if epoch%10000 == 0:
            print('[tune]')
            print(sum(abs(diff)))
        
    if k == 0:
        best1 = default_input
    elif k == 1:
        best2 = default_input
    elif k == 2:
        best3 = default_input
        
pickle.dump(best1, open('best1_0~5000.pkl', 'wb'))
pickle.dump(best2, open('best2_0~5000.pkl', 'wb'))
pickle.dump(best3, open('best3_0~5000.pkl', 'wb'))


## 順伝播でチェック
"""
best_input = default_input
x = chainer.Variable(best_input.T)
x = F.relu(model.l1(x))
x = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
y = model.l3(F.dropout(x, ratio=dropout_ratio, train=False))
y = F.softmax(y).data
"""