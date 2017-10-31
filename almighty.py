# -*- coding: utf-8 -*-


import pickle
import math
import numpy as np
from scipy.fftpack import fft
import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import random
import csv

train_dataset = []

for i in range(1, 217):
    print(i)
    for k in range(1, 31):
        fname = "tda_data/output " + str(i) + " _ " + str(k) + " .csv"
        reader = csv.reader(open(fname, 'r'))
        header = next(reader)
        list0 = []
        list1 = []
        for row in reader:
            if len(row) == 0:
                break
            if row[0] == "0":
                dummy = np.zeros((2))
                dummy[0] = float(row[1])
                dummy[1] = float(row[2])
                if len(list0) == 0:
                    list0 = dummy
                else:
                    list0 = np.c_[list0, dummy]
            else:
                dummy = np.zeros((2))
                dummy[0] = float(row[1])
                dummy[1] = float(row[2])
                if len(list1) == 0:
                    list1 = dummy
                else:
                    list1 = np.c_[list1, dummy]
        tda_diag = np.zeros((2, 1000))
        for j in range(list0.shape[1]):
            for l in range(1000):
                if list0[1, j] < 0.003 * l:
                    break
                elif 0.003 * l < list0[0, j]:
                    continue
                else:
                    tda_diag[0, l] += 1
        for j in range(list1.shape[1]):
            for l in range(1000):
                if list1[1, j] < 0.003 * l:
                    break
                elif 0.003 * l < list1[0, j]:
                    continue
                else:
                    tda_diag[1, l] += 1
        train_dataset.append(tda_diag)
    
    if i == 72:
        pickle.dump(train_dataset, open("train_dataset_tda_normal.pkl", "wb"))
        train_dataset = []
    if i == 144:
        pickle.dump(train_dataset, open("train_dataset_tda_inner.pkl", "wb"))
        train_dataset = []
    if i == 216:
        pickle.dump(train_dataset, open("train_dataset_tda_mis.pkl", "wb"))
        train_dataset = []
        

"""
# 1変数時系列データを2次元空間に埋め込み
# 周期は48Hz,サンプリングレート20000Hzなので417フレーム飛ばし
with open('train_dataset_0905.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
    
for i in range(500, 1000):
    fname = 'consequance' + str(i) + '.jpg'
    plt.figure()
    plt.plot(train_dataset[0 : i, 144], train_dataset[1 : i + 1, 144])
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.savefig(fname)
    plt.close()

"""

"""
xp = np
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

with open('test_dataset_0822.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
    
batch, output = make_batch_test(test_dataset)

for i in range(48):
    fname = 'frequency' + str(i) + '0~5000.jpg'
    plt.figure()
    plt.plot(range(100), batch[i*4, :])
    plt.xticks([0,20,40,60,80,100], ['0','1000','2000','3000','4000','5000'])
    plt.savefig(fname)

"""  
    
"""
for i in range(48):
    xx = range(100)
    fname = 'original_test_data' + str(i) + '.png'
    plt.figure()
    plt.plot(xx, batch[i, :])
    plt.savefig(fname)


xp = np
dropout_ratio = 0.1
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
        new[i] = calRms(sample[5*i:5*i+7])
    return new
def make_batch_test(dataset, batchSize=48):
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
        output[i] = i/16
    return batch, output


# 100回テストして精度見るやつ
with open('test_dataset_0713.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
model = pickle.load(open('dnn_0713_model_2kind_rowrowrowrow.pkl', 'rb'))
sums = 0
for i in range(100):
    batch, output = make_batch_test(test_dataset)
    x = chainer.Variable(batch)
    t = chainer.Variable(output)
    x = F.relu(model.l1(x))
    x = F.relu(model.l2(F.dropout(x, ratio=dropout_ratio, train=False)))
    y = model.l3(F.dropout(x, ratio=dropout_ratio, train=False))
    acc = F.accuracy(y, t).data
    sums += acc

print(sums)
"""
"""
# 重み可視化するやつ
weight2 = model.l2.W.data
weight1 = model.l1.W.data
# vec = model.l3.W.data[2, :]
# mapping = np.dot(weight1.T, np.dot(weight2.T, vec))
for i in range(3):
    vec = weight2[i, :]
    mapping = np.dot(weight1.T, vec)
    plt.figure()
    xx = range(100)
    plt.plot(xx, mapping)
    plt.xticks([0,20,40,60,80,100], ['0','100','200','300','400','500'])
    fname = 'mapping' + str(i) + '.png'
    plt.savefig(fname)
"""

"""
weight2 = model.conv2.W.data
weight1 = model.conv1.W.data

for wnum in range(50):
    w = weight2[wnum, :, :, 0]
    
    mapping = np.zeros((20, 10))
    mapping = np.array(mapping, dtype=np.float32)
    for i in range(5):
        mapping[:, 2*i] = w[:, i]
        
    mapping2 = np.zeros((14))
    mapping2 = np.array(mapping2, dtype=np.float32)
    
    for k in range(20):
        ww = weight1[k, 0, :, 0]
        for l in range(10):
            mapping2[l:l+5] += ww*mapping[k, l]
    xx = range(14)
    plt.plot(xx, mapping2)
    fname = 'allweight.png'
    plt.savefig(fname)

"""

"""
xp = np
dropout_ratio = 0.2

with open('test_dataset_0706.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

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

# バッチ作成(test用)
def make_batch_test(dataset, code):
    batch = xp.zeros((1, 100))
    batch = xp.array(batch, dtype=xp.float32)
    index = random.randint(0, 18999)
    sample = dataset[index:index+1000, code]
    sample = np.array(sample, dtype=xp.float32)
    sample = processing_data(sample)
    batch[0, :] = sample
    return batch

fname = 'dnn_0707_model_second.pkl'
model = pickle.load(open(fname, 'rb'))
for i in range(24):
    batch = make_batch_test(test_dataset, i)
    x = chainer.Variable(batch)
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(F.dropout(h1, ratio=dropout_ratio, train=False)))
    y = model.l3(F.dropout(h2, ratio=dropout_ratio, train=False))
    # y = F.softmax(y).data
    plt.imshow(y.data)
    fname = 'out_' + str(i)
    plt.savefig(fname)


xp = np

with open('test_dataset.pkl', 'rb') as f:
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
def make_batch(dataset, batchSize, num):
    batch = xp.zeros((batchSize, 1, 100, 32))
    batch = xp.array(batch, dtype=xp.float32)
    for i in range(batchSize):
        sample = dataset[1000:2000, num]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, 0, :, 0] = sample
    return batch

fname = 'model/cnn_auto_encoder_model9900.pkl'
model = pickle.load(open(fname, 'rb'))
batch = make_batch(normal_dataset, 1, 158)
x = chainer.Variable(batch)
t = x
h = F.relu(model.conv1(x))
h = F.max_pooling_2d(h, ksize=2, cover_all=False)
h = F.relu(model.conv2(h))

xx = range(100)
plt.plot(xx, batch[0,0,:,0])

"""

"""

h = F.max_pooling_2d(h, ksize=2, cover_all=False)
h = F.relu(model.conv3(h))
h = F.relu(model.deconv1(h))
h = F.unpooling_2d(h, ksize=2, cover_all=False)
h = F.relu(model.deconv2(h))
h = F.unpooling_2d(h, ksize=2, cover_all=False)
y = F.relu(model.deconv3(h))

xx = range(100)
plt.plot(xx, y.data[0, 0, :, 0])



import pickle
import math
import numpy as np
from scipy.fftpack import fft
import chainer
import chainer.functions as F
import matplotlib.pyplot as plt

xp = np
dropout_ratio = 0.2

with open('test_dataset.pkl', 'rb') as f:
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
    batch = xp.zeros((batchSize, 100))
    batch = xp.array(batch, dtype=xp.float32)
    for i in range(batchSize):
        sample = dataset[1000:2000, 14]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, :] = sample
    return batch

fname = 'model/autoencodermodel9900.pkl'
model = pickle.load(open(fname, 'rb'))
batch = make_batch(normal_dataset, 1)
x = chainer.Variable(batch)
t = x
x = model.l1(x)
x = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.l3(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.d3(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.d2(F.dropout(x, ratio=dropout_ratio, train=False))
y = model.d1(F.dropout(x, ratio=dropout_ratio, train=False))

xx = range(100)
plt.plot(xx, y.data[0, :])


plt.plot(xx, y.data[0, 0, :, 0])
plt.ylim(8, 72)
"""