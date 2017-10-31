# -*- coding: utf-8 -*-

import csv
import numpy as np
import pickle

train_dataset = []

for i in [1,2,3,4]:
    for k in [1,2,3,4,5]:
        filename = 'sample_data\\normal_p' + str(i) + '_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]
    for k in [1,2,3,4]:
        filename = 'sample_data\\normal2_p' + str(i) + '_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]

for i in [1,2,3,4]:
    for k in [1,2,3,4,5]:
        filename = 'sample_data\\normal_p' + str(i) + '_n_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]
    for k in [1,2,3,4]:
        filename = 'sample_data\\normal2_p' + str(i) + '_n_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]

for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9]:
        filename = 'sample_data\\inner2_p' + str(i) + '_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]
            
for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9]:
        filename = 'sample_data\\inner2_p' + str(i) + '_n_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]
            
for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9]:
        filename = 'sample_data\\mis2_p' + str(i) + '_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]
for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9]:
        filename = 'sample_data\\mis2_p' + str(i) + '_n_000' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(train_dataset) == 0:
            train_dataset = sound
        else:
            train_dataset = np.c_[train_dataset,sound]
            
test_dataset = []

for i in [1,2,3,4]:
    for k in [10]:
        filename = 'sample_data\\normal2_p' + str(i) + '_n_00' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]
for i in [1,2,3,4]:
    for k in [10]:
        filename = 'sample_data\\normal2_p' + str(i) + '_n_00' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]
            
for i in [1,2,3,4]:
    for k in [10]:
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
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]
            
for i in [1,2,3,4]:
    for k in [10]:
        filename = 'sample_data\\inner2_p' + str(i) + '_n_00' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]
            
for i in [1,2,3,4]:
    for k in [10]:
        filename = 'sample_data\\mis2_p' + str(i) + '_n_00' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]

for i in [1,2,3,4]:
    for k in [10]:
        filename = 'sample_data\\mis2_p' + str(i) + '_n_00' + str(k) + '.csv'
        file = open(filename, 'r')
        dataReader = csv.reader(file, delimiter='\t')
        sound = []
        for j in range(23):
            header = next(dataReader)
        for row in dataReader:
            if len(row) == 0:
                break
            sound.append(float(row[1]))
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]

pickle.dump(train_dataset, open('train_dataset_0905.pkl', 'wb'))
pickle.dump(test_dataset, open('test_dataset_0905.pkl', 'wb'))