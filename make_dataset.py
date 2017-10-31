# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:11:37 2017

@author: Share
"""
from pylab import *
import csv
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier


def make_dataset():
    #データセット作成
    in_train = []
    out_train = np.array( [] )
    
    #正常時データセット
    for i in [1,2,3,4]:
        for k in [1,2,3,4,5,6,7,8,9]:
            filename = 'sample_data\\normal_p' + str(i) + '_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\normal_p' + str(i) + '_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_train) == 0:
                in_train = sound
                out_train = np.append(out_train, 1)
            else:
                in_train = np.c_[in_train,sound]
                out_train = np.append(out_train, 1)
    for i in [1,2,3,4]:
        for k in [1,2,3,4,5,6,7,8,9]:
            filename = 'sample_data\\normal_p' + str(i) + '_n_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\normal_p' + str(i) + '_n_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_train) == 0:
                in_train = sound
                out_train = np.append(out_train, 1)
            else:
                in_train = np.c_[in_train,sound]
                out_train = np.append(out_train, 1)
    
    #異常時データセット１
    for i in [1,2,3,4]:
        for k in [1,2,3,4,5,6,7,8,9]:
            filename = 'sample_data\\inner1_p' + str(i) + '_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_train) == 0:
                in_train = sound
                out_train = np.append(out_train, 2)
            else:
                in_train = np.c_[in_train,sound]
                out_train = np.append(out_train, 2)
    for i in [1,2,3,4]:
        for k in [1,2,3,4,5,6,7,8,9]:
            filename = 'sample_data\\inner1_p' + str(i) + '_n_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_n_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_train) == 0:
                in_train = sound
                out_train = np.append(out_train, 2)
            else:
                in_train = np.c_[in_train,sound]
                out_train = np.append(out_train, 2)
    
    
    #異常時データセット２
    for i in [1,2,3,4]:
        for k in [1,2,3,4,5,6,7,8,9]:
            filename = 'sample_data\\inner1_p' + str(i) + '_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_train) == 0:
                in_train = sound
                out_train = np.append(out_train, 3)
            else:
                in_train = np.c_[in_train,sound]
                out_train = np.append(out_train, 3)
    for i in [1,2,3,4]:
        for k in [1,2,3,4,5,6,7,8,9]:
            filename = 'sample_data\\inner1_p' + str(i) + '_n_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_n_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_train) == 0:
                in_train = sound
                out_train = np.append(out_train, 3)
            else:
                in_train = np.c_[in_train,sound]
                out_train = np.append(out_train, 3)
    
    #テストデータセット作成
    in_test = []
    out_test = np.array( [] )
    #正常時データセット
    for i in [1,2,3,4]:
        for k in [10]:
            filename = 'sample_data\\normal_p' + str(i) + '_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\normal_p' + str(i) + '_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_test) == 0:
                in_test = sound
                out_test = np.append(out_test, 1)
            else:
                in_test = np.c_[in_test,sound]
                out_test = np.append(out_test, 1)
    for i in [1,2,3,4]:
        for k in [10]:
            filename = 'sample_data\\normal_p' + str(i) + '_n_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\normal_p' + str(i) + '_n_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_test) == 0:
                in_test = sound
                out_test = np.append(out_test, 1)
            else:
                in_test = np.c_[in_test,sound]
                out_test = np.append(out_test, 1)
    
    #異常時データセット１
    for i in [1,2,3,4]:
        for k in [10]:
            filename = 'sample_data\\inner1_p' + str(i) + '_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_test) == 0:
                in_test = sound
                out_test = np.append(out_test, 2)
            else:
                in_test = np.c_[in_test,sound]
                out_test = np.append(out_test, 2)
    for i in [1,2,3,4]:
        for k in [10]:
            filename = 'sample_data\\inner1_p' + str(i) + '_n_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_n_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_test) == 0:
                in_test = sound
                out_test = np.append(out_test, 2)
            else:
                in_test = np.c_[in_test,sound]
                out_test = np.append(out_test, 2)
    
    
    #異常時データセット２
    for i in [1,2,3,4]:
        for k in [10]:
            filename = 'sample_data\\inner1_p' + str(i) + '_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_test) == 0:
                in_test = sound
                out_test = np.append(out_test, 3)
            else:
                in_test = np.c_[in_test,sound]
                out_test = np.append(out_test, 3)
    for i in [1,2,3,4]:
        for k in [10]:
            filename = 'sample_data\\inner1_p' + str(i) + '_n_000' + str(k) + '.csv'
            if k == 10:
                filename = 'sample_data\\inner1_p' + str(i) + '_n_00' + str(k) + '.csv'
            file = open(filename, 'r')
            dataReader = csv.reader(file, delimiter='\t')
            sound = []
            for j in range(23):
                header = next(dataReader)
            for row in dataReader:
                if len(row) == 0:
                    break
                sound.append(float(row[1]))
            if len(in_test) == 0:
                in_test = sound
                out_test = np.append(out_test, 3)
            else:
                in_test = np.c_[in_test,sound]
                out_test = np.append(out_test, 3)
    in_train = in_train.T
    out_train = out_train.T
    in_test = in_test.T
    out_test = out_test.T

    x_train = []
    y_train = np.array( [] )
    x_test = []
    y_test = np.array( [] )
    
    for i in range(500):
        if i%100 == 0:
            print(i)
        if i == 0:
            x_train = in_train[:,i:i+200]
            y_train = out_train
        else:
            x_train = np.r_[x_train,in_train[:,i:i+200]]
            y_train = np.append(y_train, out_train)
    for i in range(500):
        if i%100 == 0:
            print(i)
        if i == 0:
            x_test = in_test[:,i:i+200]
            y_test = out_test
        else:
            x_test = np.r_[x_test,in_test[:,i:i+200]]
            y_test = np.append(y_test, out_test)
    
    x_train=np.array(x_train,dtype=np.float32)
    x_test=np.array(x_test,dtype=np.float32)
    y_train=np.array(y_train,dtype=np.int32)
    y_test=np.array(y_test,dtype=np.int32)

    
    return x_train, y_train, x_test, y_test
    