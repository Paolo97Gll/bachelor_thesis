#!/usr/bin/env python3
# coding: utf-8

# SVC: C-Support Vector Classification
# REAL LIFE TEST: test the algorithm in a real-life application i.e., scan an entire OD.


#######################################################################


# Scientific computing
import numpy as np
import pandas as pd

# Plot
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#sns.set_context('paper')

# Machine Learning
# Model
from sklearn.svm import SVC
# Ensemble model
from sklearn.ensemble import BaggingClassifier

# Other
import time


#######################################################################


# Reset out file
with open('ris/RL_test.txt', mode='w') as f:
    pass

# Load train data
first_cycle = True
with pd.HDFStore('../../classification/ris/OUT-classified-merged.h5', mode='r') as in_data:
    for group in ['GLITCH', 'NO_GLITCH']:
        if first_cycle == True:
            data = np.array(in_data[group].to_numpy())
            if group == 'GLITCH':
                target = np.ones(len(data))
            elif group == 'NO_GLITCH':
                target = np.zeros(len(data))
            else:
                print("ERROR.")
            first_cycle = False
        else:
            data = np.concatenate((data, in_data[group].to_numpy()))
            if group == 'GLITCH':
                target = np.concatenate((target, np.ones(len(in_data[group].to_numpy()))))
            elif group == 'NO_GLITCH':
                target = np.concatenate((target, np.zeros(len(in_data[group].to_numpy()))))
            else:
                print("ERROR.")
    data = np.concatenate((data, in_data['MULTI_GLITCH'].to_numpy()))
    target = np.concatenate((target, np.ones(len(in_data['MULTI_GLITCH'].to_numpy()))))
train_data = data
train_target = target
del data, target

# Set test data info
filename = '~/data/OUT-cleaned.h5'
OPERATING_DAY_LIST = ['091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105']
DETECTOR_LIST = ['143-5', '143-6', '143-7']


#######################################################################
# Normal model


with open('ris/RL_test.txt', mode='a') as f:
    print('##### Normal model #####', file=f)

T1 = time.time()

best_kernel = 'rbf'
best_gamma = 0.0151
best_C = 1.45

clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C, probability=True)
clf.fit(train_data, train_target)

t_list = []
for OD in OPERATING_DAY_LIST:
    for detector in DETECTOR_LIST:
        
        print('OD:', OD, '- Detector:', detector)
        
        curr_df = OD + '/' + detector
        with pd.HDFStore(filename, mode='r') as in_data:
            data_df = in_data[curr_df]
        n_sequences = data_df.index[::100].shape[0]

        t = 0
        for i in range(0, n_sequences-1):
            d = data_df.iloc[i*100 : i*100 + 100].to_numpy().transpose()
            t_b = time.time()
            clf.predict_proba(d)
            t_e = time.time()
            t += (t_e - t_b)
            #print(str(int(i/n_sequences*10000)/100) + '%', end='\r')
        #print('Time [s]:', t)
        
        t_list.append(t)
        with open('ris/RL_test.txt', mode='a') as f:
            print('OD ' + OD + ', detector ' + detector + ' [s]:', t, file=f)

T2 = time.time()

with open('ris/RL_test.txt', mode='a') as f:
    print('Model time [s]:', np.sum(t_list), file=f)
    print('Total time [s]:', T2 - T1, file=f)


#######################################################################
# Bagging model


with open('ris/RL_test.txt', mode='a') as f:
    print('##### Bagging model #####', file=f)

T1 = time.time()

n_estimators = 4
max_samples = 0.95
best_kernel = 'rbf'
best_gamma = 0.0151
best_C = 1.45

clf = BaggingClassifier(SVC(kernel=best_kernel, gamma=best_gamma, C=best_C), n_estimators=n_estimators, max_samples=max_samples, n_jobs=-1)
clf.fit(train_data, train_target)

t_list = []
for OD in OPERATING_DAY_LIST:
    for detector in DETECTOR_LIST:
        
        print('OD:', OD, '- Detector:', detector)
        
        curr_df = OD + '/' + detector
        with pd.HDFStore(filename, mode='r') as in_data:
            data_df = in_data[curr_df]
        n_sequences = data_df.index[::100].shape[0]

        t = 0
        for i in range(0, n_sequences-1):
            d = data_df.iloc[i*100 : i*100 + 100].to_numpy().transpose()
            t_b = time.time()
            clf.predict_proba(d)
            t_e = time.time()
            t += (t_e - t_b)
            #print(str(int(i/n_sequences*10000)/100) + '%', end='\r')
        #print('Time [s]:', t)
        
        t_list.append(t)
        with open('ris/RL_test.txt', mode='a') as f:
            print('OD ' + OD + ', detector ' + detector + ' [s]:', t, file=f)

T2 = time.time()

with open('ris/RL_test.txt', mode='a') as f:
    print('Model time [s]:', np.sum(t_list), file=f)
    print('Total time [s]:', T2 - T1, file=f)


#######################################################################
# Sorted model


with open('ris/RL_test.txt', mode='a') as f:
    print('##### Sorted model #####', file=f)

T1 = time.time()

train_data_s = np.sort(train_data, axis=1)

best_kernel = 'linear'
best_C = 0.15

clf = SVC(kernel=best_kernel, C=best_C, probability=True)
clf.fit(train_data_s, train_target)

t_list = []
for OD in OPERATING_DAY_LIST:
    for detector in DETECTOR_LIST:
        
        print('OD:', OD, '- Detector:', detector)
        
        curr_df = OD + '/' + detector
        with pd.HDFStore(filename, mode='r') as in_data:
            data_df = in_data[curr_df]
        n_sequences = data_df.index[::100].shape[0]

        t = 0
        for i in range(0, n_sequences-1):
            d = np.sort(data_df.iloc[i*100 : i*100 + 100].to_numpy().transpose(), axis=1)
            t_b = time.time()
            clf.predict_proba(d)
            t_e = time.time()
            t += (t_e - t_b)
            #print(str(int(i/n_sequences*10000)/100) + '%', end='\r')
        #print('Time [s]:', t)
        
        t_list.append(t)
        with open('ris/RL_test.txt', mode='a') as f:
            print('OD ' + OD + ', detector ' + detector + ' [s]:', t, file=f)

T2 = time.time()

with open('ris/RL_test.txt', mode='a') as f:
    print('Model time [s]:', np.sum(t_list), file=f)
    print('Total time [s]:', T2 - T1, file=f)