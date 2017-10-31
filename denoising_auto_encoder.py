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

# 学習データの読み込み
with open('normal_dataset.pkl', 'rb') as f:
    normal_dataset = pickle.load(f)

# 与えるノイズの乱数作成時、1を何等分するか
break_num = 5
# 学習データをランダムに破壊する
roopX, roopY = normal_dataset.shape
normal_dataset_input = np.zeros((roopX, roopY))
for x in range(roopX):
    for y in range(roopY):
        index = random.randint(0, break_num)
        diff = (random.randint(0, index)*(random.random() - 0.5)*2)/break_num
        normal_dataset_input[x, y] = normal_dataset[x, y]*diff + normal_dataset[x, y]

xp = np
in_units = 100
hidden_units1 = 64
hidden_units2 = 16
hidden_units3 = 8
training_epochs = 10000
batch_size = 72
dropout_ratio = 0.5
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
def make_batch(dataset_in, dataset_out, batchSize):
    batch = xp.zeros((batchSize, 100))
    batch = xp.array(batch, dtype=xp.float32)
    out = xp.zeros((batchSize, 100))
    out = xp.array(out, dtype=xp.float32)
    for i in range(batchSize):
        index = random.randint(0, 18999)
        sample_in = dataset_in[index:index+1000, i]
        sample_in = np.array(sample_in, dtype=xp.float32)
        sample_in = processing_data(sample_in)
        batch[i, :] = sample_in
        sample_out = dataset_out[index:index+1000, i]
        sample_out = np.array(sample_out, dtype=xp.float32)
        sample_out = processing_data(sample_out)
        out[i, :] = sample_out
    return batch, out

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
    batch, output = make_batch(normal_dataset_input, normal_dataset, batch_size)
    x = chainer.Variable(batch)
    t = chainer.Variable(output)
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
    batch, output = make_batch(normal_dataset_input, normal_dataset, batch_size)
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
    batch, output = make_batch(normal_dataset_input, normal_dataset, batch_size)
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
    batch, output = make_batch(normal_dataset_input, normal_dataset, batch_size)
    x = chainer.Variable(batch)
    t = chainer.Variable(output)
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

# 以下、再構築誤差取得用スクリプト
# テストデータの読み込み

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
    

