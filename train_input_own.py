# -*- coding: utf-8 -*-
"""
隠れ層1層のニューラルネット活性化関数relu→softmaxの学習済みモデル
で、目的の出力値に最も反応する入力を学習させる
"""

import pickle
import numpy as np
import chainer.functions as F
import chainer
import matplotlib.pyplot as plt


model = pickle.load(open('1layer_3kind_drop0.02_0_1000Hz_0822.pkl', 'rb'))

input_nodes = 100
hidden_nodes = 32
output_nodes = 3
output_cluster = 0
learning_rate = 0.1
epochs = 300000
dropout_ratio = 0.4

for k in [0, 1, 2]:
    output = np.zeros((output_nodes))
    output = np.array(output, dtype=np.float32)
    output = output[:, np.newaxis]
    
    output[k] = 1
    
    default_input = np.ones((input_nodes))
    default_input = np.array(default_input, dtype=np.float32)
    default_input = default_input[:, np.newaxis]
    
    w1 = model.l1.W.data
    w2 = model.l2.W.data
    
    for epoch in range(epochs):
        y = np.dot(w1, default_input)
        del_y = np.ones((hidden_nodes))
        del_y = del_y[:, np.newaxis]
        for i in range(hidden_nodes):
            if y[i] < 0:
                y[i] = 0
                del_y[i] = 0
        y = np.dot(w2, y)
        y = chainer.Variable(y.T)
        y = F.softmax(y).data.T
        
        diff = y - output
        diff = np.dot(w2.T, diff)
        
        default_input -= learning_rate * np.dot(w1.T, diff * del_y)
        
        for i in range(input_nodes):
            if default_input[i] < 0:
                default_input[i] = 0
    
        print(sum(abs(diff)))
        
    if k == 0:
        best1 = default_input
    elif k == 1:
        best2 = default_input
    elif k == 2:
        best3 = default_input

pickle.dump(best1, open('best1_0_1000.pkl', 'wb'))
pickle.dump(best2, open('best2_0_1000.pkl', 'wb'))
pickle.dump(best3, open('best3_0_1000.pkl', 'wb'))

## 順伝播でチェック
"""
best_input = default_input
x = chainer.Variable(best_input.T)
x = F.relu(model.l1(x))
y = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
y = F.softmax(y).data
"""