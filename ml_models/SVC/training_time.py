#!/usr/bin/env python3
# coding: utf-8

# SVC: C-Support Vector Classification
# Score SVC alglorithms (no and yes multi glitches) with simple data, data augmentation.


#######################################################################


# Reading files
import toml

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
# Model
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

# Other
import time
import requests
import threading


#######################################################################


# Multithreading
maxthreads = 4
sema = threading.Semaphore(value=maxthreads)
c = threading.Condition()

# Bagging Classifier parameters
n_estimators = 4
max_samples = 0.95

# Reset out file
with open('ris/OUT-time_alglorithms.txt', mode='w') as f:
    pass

telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#######################################################################
# NO MULTI GLITCH
#######################################################################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start no multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Start no multi glitch part.', file=f)

# Load data
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

# Best training parameters
best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8


############
# Standard #
############


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Standard training.', file=f)

clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
t_b = time.time()
clf.fit(data, target)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#####################
# Data augmentation #
#####################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Start data augmentation part.', file=f)

data_aug, target_aug = data, target
data_aug = np.concatenate((data_aug, -data))
target_aug = np.concatenate((target_aug, target))
for j in range(1,100):
    data_aug = np.concatenate((data_aug, np.roll(data, j, axis=1)))
    data_aug = np.concatenate((data_aug, -np.roll(data, j, axis=1)))
    target_aug = np.concatenate((target_aug, target))
    target_aug = np.concatenate((target_aug, target))


#################
# Simple training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Standard training.', file=f)

clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
t_b = time.time()
clf.fit(data_aug, target_aug)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#############################
# Bagging Classifier training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start Bagging Classifier training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Bagging Classifier training.', file=f)

clf = BaggingClassifier(SVC(kernel=best_kernel, gamma=best_gamma, C=best_C), n_estimators=n_estimators, max_samples=max_samples, n_jobs=-1)
t_b = time.time()
clf.fit(data_aug, target_aug)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End Bagging Classifier training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('End data augmentation part.', file=f)
    

###############
# Sorted data #
###############


data_s = np.sort(data, axis=1)

best_kernel_s = 'linear'
best_C_s = 0.117

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start sorted training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Sorted training.', file=f)

clf = SVC(kernel=best_kernel_s, C=best_C_s)
t_b = time.time()
clf.fit(data_s, target)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End sorted training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#######
# END #
#######


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End no multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('End no multi glitch part.', file=f)


#######################################################################
# YES MULTI GLITCH
#######################################################################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start yes multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('\nStart yes multi glitch part.', file=f)

# Load data
with pd.HDFStore('../../classification/ris/OUT-classified-merged.h5', mode='r') as in_data:
    data = np.concatenate((data, in_data['MULTI_GLITCH'].to_numpy()))
    target = np.concatenate((target, np.ones(len(in_data['MULTI_GLITCH'].to_numpy()))))
                
# Best training parameters
best_kernel = 'rbf'
best_gamma = 0.0151
best_C = 1.45


############
# Standard #
############


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Standard training.', file=f)

clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
t_b = time.time()
clf.fit(data, target)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#####################
# Data augmentation #
#####################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Start data augmentation part.', file=f)

data_aug, target_aug = data, target
data_aug = np.concatenate((data_aug, -data))
target_aug = np.concatenate((target_aug, target))
for j in range(1,100):
    data_aug = np.concatenate((data_aug, np.roll(data, j, axis=1)))
    data_aug = np.concatenate((data_aug, -np.roll(data, j, axis=1)))
    target_aug = np.concatenate((target_aug, target))
    target_aug = np.concatenate((target_aug, target))


#################
# Simple training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Standard training.', file=f)

clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
t_b = time.time()
clf.fit(data_aug, target_aug)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End standard training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#############################
# Bagging Classifier training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start Bagging Classifier training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Bagging Classifier training.', file=f)

clf = BaggingClassifier(SVC(kernel=best_kernel, gamma=best_gamma, C=best_C), n_estimators=n_estimators, max_samples=max_samples, n_jobs=-1)
t_b = time.time()
clf.fit(data_aug, target_aug)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End Bagging Classifier training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('End data augmentation part.', file=f)
    

###############
# Sorted data #
###############


data_s = np.sort(data, axis=1)

best_kernel_s = 'linear'
best_C_s = 0.117

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start sorted training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Sorted training.', file=f)

clf = SVC(kernel=best_kernel_s, C=best_C_s)
t_b = time.time()
clf.fit(data_s, target)
t_e = time.time()

# Print final score
with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End sorted training.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#######
# END #
#######


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End yes multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('End yes multi glitch part.', file=f)