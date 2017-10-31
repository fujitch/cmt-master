# -*- coding: utf-8 -*-

import numpy as np
import chainer.functions as F
import chainer.links as L
import chainer
from scipy.fftpack import fft
import pickle
import math
import random
from chainer import optimizers

with open('train_dataset_tda_normal.pkl', 'rb') as f:
    dataset_normal = pickle.load(f)
with open('train_dataset_tda_inner.pkl', 'rb') as f:
    dataset_inner = pickle.load(f)
with open('train_dataset_tda_mis.pkl', 'rb') as f:
    dataset_mis = pickle.load(f)
    
train_dataset_normal = []
train_dataset_inner = []
train_dataset_mis = []

test_dataset_normal = []
test_dataset_inner = []
test_dataset_mis = []

for i in range(2160):
    if i%30 == 0:
        test_dataset_normal.append(dataset_normal[i])
        test_dataset_inner.append(dataset_inner[i])
        test_dataset_mis.append(dataset_mis[i])
    else:
        train_dataset_normal.append(dataset_normal[i])
        train_dataset_inner.append(dataset_inner[i])
        train_dataset_mis.append(dataset_mis[i])

del(dataset_normal)
del(dataset_inner)
del(dataset_mis)

xp = np

in_units = 2000
hidden_units = 500
out_units = 3
training_epochs = 15000
batch_size = 216
dropout_ratio = 0.01
display_epoch = 100

# バッチ作成
def make_batch():
    batch = xp.zeros((batch_size, 2000))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batch_size))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batch_size):
        index = random.randint(0, 2087)
        if i/72 == 0:
            sample = train_dataset_normal[index]
            sample = xp.reshape(sample, (2000))
            batch[i, :] = sample
            output[i] = 0
        elif i/72 == 1:
            sample = train_dataset_inner[index]
            sample = xp.reshape(sample, (2000))
            batch[i, :] = sample
            output[i] = 1
        elif i/72 == 2:
            sample = train_dataset_mis[index]
            sample = xp.reshape(sample, (2000))
            batch[i, :] = sample
            output[i] = 2
    return batch, output

# バッチ作成(test用)
def make_batch_test():
    batch = xp.zeros((batch_size, 2000))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batch_size))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batch_size):
        index = i%72
        if i/72 == 0:
            sample = test_dataset_normal[index]
            sample = xp.reshape(sample, (2000))
            batch[i, :] = sample
            output[i] = 0
        elif i/72 == 1:
            sample = test_dataset_inner[index]
            sample = xp.reshape(sample, (2000))
            batch[i, :] = sample
            output[i] = 1
        elif i/72 == 2:
            sample = test_dataset_mis[index]
            sample = xp.reshape(sample, (2000))
            batch[i, :] = sample
            output[i] = 2
    return batch, output

# DNNモデルの作成
model = chainer.FunctionSet(l1=L.Linear(in_units, hidden_units),
                            l2=L.Linear(hidden_units, out_units))

# 重みの初期化
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)

# model = pickle.load(open('dnn_0713_model_3kind_all_row_row_row_fre.pkl', 'rb'))
# optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

## テスト用データをランダム生成しておく
test_batch, test_output = make_batch_test()

accMatrix = np.zeros((training_epochs))
sumsMatrix = np.zeros((training_epochs))

# fine-chooning
for epoch in range(training_epochs):
    batch, output = make_batch()
    optimizer.zero_grads()
    x = chainer.Variable(batch)
    t = chainer.Variable(output)
    x = F.relu(model.l1(x))
    y = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
    optimizer.update()
    accMatrix[epoch] = F.accuracy(y, t).data
    if (epoch%100 == 0):
        print(
            "Fin[{j}]training loss:\t{i}\t acc:\t{k}".format(
                j=epoch, 
                i=loss.data/(in_units - 1),
                k=F.accuracy(y, t).data
            )
        )
    x = chainer.Variable(test_batch)
    t = chainer.Variable(test_output)
    x = F.relu(model.l1(x))
    y = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
    sumsMatrix[epoch] = F.accuracy(y, t).data
pickle.dump(sumsMatrix, open('sumsMatrix_nn_tda.pkl', 'wb'))
pickle.dump(accMatrix, open('accMatrix_nn_tda.pkl', 'wb'))        
pickle.dump(model, open('model_nn_tda.pkl', 'wb'))
            
# 以下、再構築誤差取得用スクリプト
# テストデータの読み込み
"""
with open('test_dataset_0713.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
    
batch, output = make_batch_test(test_dataset)
x = chainer.Variable(batch)
t = chainer.Variable(output)
x = F.relu(model.l1(x))
y = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
y = F.softmax(y).data

"""