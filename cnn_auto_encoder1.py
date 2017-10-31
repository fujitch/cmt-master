# -*- coding: utf-8 -*-

import numpy as np
import pickle
import math
from scipy.fftpack import fft
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

xp = np
TRAINING_EPOCHS = 10000
MINI_BATCH_SIZE = 72
dropout_ratio = 0.5

# 学習データの読み込み
with open('normal_dataset.pkl', 'rb') as f:
    normal_dataset = pickle.load(f)
    
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
    batch = xp.zeros((batchSize, 1, 100, 5))
    batch = xp.array(batch, dtype=xp.float32)
    for i in range(batchSize):
        index = random.randint(0, 18999)
        sample = dataset[index:index+1000, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, 0, :, 0] = sample
    return batch

# バッチ作成(test用)
def make_batch_test(dataset, code, batchSize=80):
    batch = xp.zeros((batchSize, 1, 100, 5))
    batch = xp.array(batch, dtype=xp.float32)
    if code == 0:
        for i in range(batchSize):
            index = random.randint(0, 18999)
            sample = dataset[index:index+1000, i/10]
            sample = np.array(sample, dtype=xp.float32)
            sample = processing_data(sample)
            batch[i, 0, :, 0] = sample
    elif code == 1:
        for i in range(batchSize):
            index = random.randint(0, 18999)
            sample = dataset[index:index+1000, i+8]
            sample = np.array(sample, dtype=xp.float32)
            sample = processing_data(sample)
            batch[i, 0, :, 0] = sample
    elif code == 2:
        for i in range(batchSize):
            index = random.randint(0, 18999)
            sample = dataset[index:index+1000, i+88]
            sample = np.array(sample, dtype=xp.float32)
            sample = processing_data(sample)
            batch[i, 0, :, 0] = sample
    return batch

model = chainer.FunctionSet(conv1=L.Convolution2D(1, 5, 5),
                            deconv1=L.Deconvolution2D(5, 1, 5))

for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    
optimizer = optimizers.Adam()
optimizer.setup(model)

#トレーニング
for epoch in range(TRAINING_EPOCHS):
    imageBatch = make_batch(normal_dataset, MINI_BATCH_SIZE)
    optimizer.zero_grads()
    x = chainer.Variable(imageBatch)
    t = chainer.Variable(imageBatch)
    h = F.relu(model.conv1(x))
    y = F.relu(model.deconv1(F.dropout(h, ratio=dropout_ratio, train=True)))
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Pre1[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(imageBatch.shape[0] - 1)
            )
        )
    if (epoch%100 == 0):
        fn = 'model/cnn_auto_encoder_model_1layer' + str(epoch) + '.pkl'
        pickle.dump(model, open(fn, 'wb'))
            

            
            
pickle.dump(model, open('cnn_auto_encoder_model_1layer.pkl', 'wb'))

# 以下、再構築誤差取得用スクリプト
# テストデータの読み込み

with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
    
batch = make_batch_test(test_dataset, 2)
x = chainer.Variable(batch)
t = x
h = F.relu(model.conv1(x))
y = F.relu(model.deconv1(h))
loss = y.data - batch

loss_list = []
for i in range(80):
    mat = loss[i, 0, :, :]
    mat = np.power(mat,2)
    mat = sum(sum(mat))
    loss_list.append(mat)
    

