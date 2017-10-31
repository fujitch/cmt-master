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
PRE_TRAINING_EPOCHS = 10
TRAINING_EPOCHS = 100
MINI_BATCH_SIZE = 216
dropout_ratio = 0.5

# 学習データの読み込み
with open('train_dataset_0706.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

# バッチ作成
def make_batch(dataset, batchSize):
    batch = xp.zeros((batchSize, 1, 40, 40))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batchSize))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batchSize):
        index = random.randint(0, 18399)
        sample = dataset[index:index+1600, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = sample.reshape(40, 40)
        batch[i, 0, :, :] = sample
        if (i/72) == 0:
            output[i] = 0
        else:
            output[i] = 1
    return batch, output

# バッチ作成(test用)
def make_batch_test(dataset, batchSize=24):
    batch = xp.zeros((batchSize, 1, 40, 40))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batchSize))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batchSize):
        index = random.randint(0, 18399)
        sample = dataset[index:index+1600, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = sample.reshape(40, 40)
        batch[i, 0, :, :] = sample
        if (i/8) == 0:
            output[i] = 0
        else:
            output[i] = 1
    return batch, output

model = chainer.FunctionSet(conv1=F.Convolution2D(3, 20, 5),
                            conv2=F.Convolution2D(20, 50, 5),
                            conv3=F.Convolution2D(50, 100, 5),
                            l1=F.Linear(900, 128),
                            l2=F.Linear(128, 16),
                            l3=F.Linear(16, 2),
                            d1=F.Linear(128, 900),
                            d2=F.Linear(16, 128),
                            deconv1=L.Deconvolution2D(100, 50, 5),
                            deconv2=L.Deconvolution2D(50, 20, 5),
                            deconv3=L.Deconvolution2D(20, 1, 5))

for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    
optimizer = optimizers.Adam()
optimizer.setup(model)

#プレトレーニング
for epoch in range(PRE_TRAINING_EPOCHS):
    imageBatch, output = make_batch(train_dataset, MINI_BATCH_SIZE)
    optimizer.zero_grads()
    x = chainer.Variable(imageBatch)
    t = chainer.Variable(imageBatch)
    h = F.relu(model.conv1(x))
    y = F.relu(model.deconv3(h))
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
            
#プレトレーニング
for epoch in range(PRE_TRAINING_EPOCHS):
    imageBatch, output = make_batch(train_dataset, MINI_BATCH_SIZE)
    optimizer.zero_grads()
    x = chainer.Variable(imageBatch)
    x = F.relu(model.conv1(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=False)
    x = chainer.Variable(x.data)
    t = x
    h = F.relu(model.conv2(x))
    y = F.relu(model.deconv2(h))
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Pre2[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(imageBatch.shape[0] - 1)
            )
        )
            
#プレトレーニング
for epoch in range(PRE_TRAINING_EPOCHS):
    imageBatch, output = make_batch(train_dataset, MINI_BATCH_SIZE)
    optimizer.zero_grads()
    x = chainer.Variable(imageBatch)
    x = F.relu(model.conv1(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=False)
    x = F.relu(model.conv2(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=False)
    x = chainer.Variable(x.data)
    t = x
    h = F.relu(model.conv3(x))
    y = F.relu(model.deconv1(h))
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Pre3[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(imageBatch.shape[0] - 1)
            )
        )
            
            

            
#ファインチューニング
for epoch in range(TRAINING_EPOCHS):
    imageBatch, output = make_batch(train_dataset, MINI_BATCH_SIZE)
    optimizer.zero_grads()
    x = chainer.Variable(imageBatch)
    t = chainer.Variable(output)
    x = F.relu(model.conv1(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=False)
    x = F.relu(model.conv2(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=False)
    x = F.relu(model.conv3(x))
    x = model.l1(x)
    x = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
    y = model.l3(F.dropout(x, ratio=dropout_ratio, train=True))
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
    optimizer.update()
    if (epoch%100 == 0):
        print(
            "Fin[{j}]training loss:\t{i}".format(
                j=epoch, 
                i=loss.data/(imageBatch.shape[0] - 1)
            )
        )
    if (epoch%100 == 0):
        fn = 'model/cnn_40_40_model' + str(epoch) + '.pkl'
        pickle.dump(model, open(fn, 'wb'))
            
            
pickle.dump(model, open('cnn_40_40_model.pkl', 'wb'))

# 以下、再構築誤差取得用スクリプト
# テストデータの読み込み
"""
with open('test_dataset_0706.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
    
batch, output = make_batch_test(test_dataset)
x = chainer.Variable(batch)
x = F.relu(model.conv1(x))
x = F.max_pooling_2d(x, ksize=2, cover_all=False)
x = F.relu(model.conv2(x))
x = F.max_pooling_2d(x, ksize=2, cover_all=False)
x = F.relu(model.conv3(x))
x = model.l1(x)
x = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
y = model.l3(F.dropout(x, ratio=dropout_ratio, train=True))

    
"""

