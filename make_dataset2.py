# -*- coding: utf-8 -*-

import csv
import numpy as np

normal_dataset = []

for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9]:
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
        if len(normal_dataset) == 0:
            normal_dataset = sound
        else:
            normal_dataset = np.c_[normal_dataset,sound]
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
        if len(normal_dataset) == 0:
            normal_dataset = sound
        else:
            normal_dataset = np.c_[normal_dataset,sound]
            
test_dataset = []

for i in [1,2,3,4]:
    for k in [10]:
        filename = 'sample_data\\normal_p' + str(i) + '_000' + str(k) + '.csv'
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
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]
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
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]

for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9,10]:
        filename = 'sample_data\\inner1_p' + str(i) + '_000' + str(k) + '.csv'
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
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]
for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9,10]:
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
        if len(test_dataset) == 0:
            test_dataset = sound
        else:
            test_dataset = np.c_[test_dataset,sound]

for i in [1,2,3,4]:
    for k in [1,2,3,4,5,6,7,8,9,10]:
        filename = 'sample_data\\inner2_p' + str(i) + '_000' + str(k) + '.csv'
        if k == 10:
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
    for k in [1,2,3,4,5,6,7,8,9,10]:
        filename = 'sample_data\\inner2_p' + str(i) + '_n_000' + str(k) + '.csv'
        if k == 10:
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