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

with open('train_dataset_0822.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('test_dataset_0822.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

xp = np
in_units = 100
hidden_units = 32
hidden_units2 = 8
out_units = 3
training_epochs = 5000
batch_size = 288
dropout_ratio = 0.02
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
        new[i] = calRms(sample[50*i:50*i+75])
    return new

# バッチ作成
def make_batch(dataset, batchSize):
    batch = xp.zeros((batchSize, 100))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batchSize))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batchSize):
        index = random.randint(0, 9999)
        sample = dataset[index:index+10000, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, :] = sample
        output[i] = i/96
    return batch, output

# バッチ作成(test用)
def make_batch_test(dataset, batchSize=192):
    batch = xp.zeros((batchSize, 100))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batchSize))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batchSize):
        index = random.randint(0, 9999)
        sample = dataset[index:index+10000, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, :] = sample
        output[i] = i/64
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
batches = xp.zeros((192, 100, 100))
batches = xp.array(batches, dtype=xp.float32)
outputs = xp.zeros((192, 100))
outputs = xp.array(outputs, dtype=xp.int32)
for i in range(100):
    batch, output = make_batch_test(test_dataset)
    batches[:, :, i] = batch
    outputs[:, i] = output

sumsMatrix = xp.zeros((training_epochs))
# fine-chooning
for epoch in range(training_epochs):
    batch, output = make_batch(train_dataset, batch_size)
    optimizer.zero_grads()
    x = chainer.Variable(batch)
    t = chainer.Variable(output)
    x = F.relu(model.l1(x))
    y = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Fin[{j}]training loss:\t{i}\t acc:\t{k}".format(
                j=epoch, 
                i=loss.data/(in_units - 1),
                k=F.accuracy(y, t).data
            )
        )
    sums = 0
    for i in range(100):
        batch = batches[:, :, i]
        output = outputs[:, i]
        x = chainer.Variable(batch)
        t = chainer.Variable(output)
        x = F.relu(model.l1(x))
        y = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
        acc = F.accuracy(y, t).data
        sums += acc
    sumsMatrix[epoch] = sums
#    if (epoch%1000 == 0):
#        fn = 'model/1layer_3kind_drop0.3' + str(epoch) + '.pkl'
#        pickle.dump(model, open(fn, 'wb'))
        
pickle.dump(model, open('1layer_3kind_drop0.02_0_5000Hz_0822.pkl', 'wb'))
            
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