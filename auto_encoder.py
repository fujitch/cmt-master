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

with open('normal_dataset.pkl', 'rb') as f:
    normal_dataset = pickle.load(f)

xp = np
in_units = 100
hidden_units1 = 64
hidden_units2 = 16
hidden_units3 = 8
training_epochs = 10000
batch_size = 72
dropout_ratio = 0.2
display_epoch = 100

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

# fftしてRMSを用いて短冊化する
def processing_data(sample):
    sample = sample/(calRms(sample))
    sample = fft(sample)
    sample = abs(sample)
    new = np.zeros((100))
    new = xp.array(new, dtype=xp.float32)
    for i in range(100):
        new[i] = calRms(sample[i:i+10])
    return new

# バッチ作成
def make_batch(dataset, batchSize):
    batch = xp.zeros((batchSize, 100))
    batch = xp.array(batch, dtype=xp.float32)
    for i in range(batchSize):
        index = random.randint(0, 18999)
        sample = dataset[index:index+1000, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, :] = sample
    return batch

# バッチ作成(test用)
def make_batch_test(dataset, code, batchSize=80):
    batch = xp.zeros((batchSize, 100))
    batch = xp.array(batch, dtype=xp.float32)
    if code == 0:
        for i in range(batchSize):
            index = random.randint(0, 18999)
            sample = dataset[index:index+1000, i/10]
            sample = np.array(sample, dtype=xp.float32)
            sample = processing_data(sample)
            batch[i, :] = sample
    elif code == 1:
        for i in range(batchSize):
            index = random.randint(0, 18999)
            sample = dataset[index:index+1000, i+8]
            sample = np.array(sample, dtype=xp.float32)
            sample = processing_data(sample)
            batch[i, :] = sample
    elif code == 2:
        for i in range(batchSize):
            index = random.randint(0, 18999)
            sample = dataset[index:index+1000, i+88]
            sample = np.array(sample, dtype=xp.float32)
            sample = processing_data(sample)
            batch[i, :] = sample
    return batch

# DNNモデルの作成
model = chainer.FunctionSet(l1=L.Linear(in_units, hidden_units1),
                            l2=L.Linear(hidden_units1, hidden_units2),
                            l3=L.Linear(hidden_units2, hidden_units3),
                            d1=L.Linear(hidden_units1, in_units),
                            d2=L.Linear(hidden_units2, hidden_units1),
                            d3=L.Linear(hidden_units3, hidden_units2))

# 重みの初期化
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    
# optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# pre-training
for epoch in range(training_epochs):
    optimizer.zero_grads()
    batch = make_batch(normal_dataset, batch_size)
    x = chainer.Variable(batch)
    t = chainer.Variable(batch)
    h = model.l1(x)
    y = model.d1(h)
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Pre1[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(in_units - 1)
            )
        )
            
# pre-training
for epoch in range(training_epochs):
    optimizer.zero_grads()
    batch = make_batch(normal_dataset, batch_size)
    x = chainer.Variable(batch)
    x = model.l1(x)
    t = x
    h = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
    y = model.d2(F.dropout(h, ratio=dropout_ratio, train=True))
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Pre2[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(hidden_units1 - 1)
            )
        )
            

# pre-training
for epoch in range(training_epochs):
    optimizer.zero_grads()
    batch = make_batch(normal_dataset, batch_size)
    x = chainer.Variable(batch)
    x = model.l1(x)
    x = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
    t = x
    h = model.l3(F.dropout(x, ratio=dropout_ratio, train=True))
    y = model.d3(F.dropout(h, ratio=dropout_ratio, train=True))
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Pre3[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(hidden_units2 - 1)
            )
        )

# fine-chooning
for epoch in range(training_epochs):
    optimizer.zero_grads()
    batch = make_batch(normal_dataset, batch_size)
    x = chainer.Variable(batch)
    t = x
    x = model.l1(x)
    x = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
    x = model.l3(F.dropout(x, ratio=dropout_ratio, train=True))
    x = model.d3(F.dropout(x, ratio=dropout_ratio, train=True))
    x = model.d2(F.dropout(x, ratio=dropout_ratio, train=True))
    y = model.d1(F.dropout(x, ratio=dropout_ratio, train=True))
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Fin[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(hidden_units2 - 1)
            )
        )
    if (epoch%100 == 0):
        fn = 'model/autoencodermodel' + str(epoch) + '.pkl'
        pickle.dump(model, open(fn, 'wb'))
        
pickle.dump(model, open('auto_encoder_model.pkl', 'wb'))
            
# 以下、再構築誤差取得用スクリプト
# テストデータの読み込み
"""
with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
    
batch = make_batch_test(test_dataset, 2)
x = chainer.Variable(batch)
t = x
x = model.l1(x)
x = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.l3(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.d3(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.d2(F.dropout(x, ratio=dropout_ratio, train=False))
y = model.d1(F.dropout(x, ratio=dropout_ratio, train=False))
loss = y.data - batch

loss_list = []
for i in range(80):
    mat = loss[i, :]
    mat = np.power(mat,2)
    mat = sum(mat)
    loss_list.append(mat)
"""