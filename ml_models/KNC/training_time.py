#!/usr/bin/env python3
# coding: utf-8

# RFC: Random Forest Classifier
# Training time RFC alglorithms with simple data and data augmentation.


#######################################################################


# Reading files
import toml

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
# Model
from sklearn.neighbors import KNeighborsClassifier
# Splitter Classes
from sklearn.model_selection import RepeatedStratifiedKFold

# Other
import time
import requests


#######################################################################

# Reset out file
with open('ris/OUT-time_alglorithms.txt', mode='w') as f:
    pass

telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

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
    data = np.concatenate((data, in_data['MULTI_GLITCH'].to_numpy()))
    target = np.concatenate((target, np.ones(len(in_data['MULTI_GLITCH'].to_numpy()))))

# Best training parameters
best_weights = 'uniform'
best_p = 8
best_n_neighbors = 1
best_leaf_size = 10
best_algorithm = 'ball_tree'


##########
# Simple #
##########


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start simple model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('# Simple model.', file=f)


################
# Training time


clf = KNeighborsClassifier(n_neighbors=best_n_neighbors,
                           weights=best_weights,
                           algorithm=best_algorithm,
                           leaf_size=best_leaf_size,
                           p=best_p,
                           n_jobs=-1)

# Compute training time
t_b = time.time()
clf.fit(data,target)
t_e = time.time()

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)


#####################
# Data augmentation #
#####################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('\n# Data augmentation model.', file=f)


################
# Training time


clf = KNeighborsClassifier(n_neighbors=best_n_neighbors,
                           weights=best_weights,
                           algorithm=best_algorithm,
                           leaf_size=best_leaf_size,
                           p=best_p,
                           n_jobs=-1)

# Data augmentation
data_aug, target_aug = data, target
data_aug = np.concatenate((data_aug, -data))
target_aug = np.concatenate((target_aug, target))
for j in range(1,100):
    data_aug = np.concatenate((data_aug, np.roll(data, j, axis=1)))
    data_aug = np.concatenate((data_aug, -np.roll(data, j, axis=1)))
    target_aug = np.concatenate((target_aug, target))
    target_aug = np.concatenate((target_aug, target))

# Compute training time
t_b = time.time()
clf.fit(data_aug, target_aug)
t_e = time.time()

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)


##########
# Sorted #
##########


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start sorted model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('\n# Sorted model.', file=f)

# Sort data
data.sort(axis=1)

# Best training parameters
best_weights = 'uniform'
best_p = 3
best_n_neighbors = 1
best_leaf_size = 10
best_algorithm = 'ball_tree'


################
# Training time


clf = KNeighborsClassifier(n_neighbors=best_n_neighbors,
                           weights=best_weights,
                           algorithm=best_algorithm,
                           leaf_size=best_leaf_size,
                           p=best_p,
                           n_jobs=-1)

# Compute training time
t_b = time.time()
clf.fit(data,target)
t_e = time.time()

with open('ris/OUT-time_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)


######################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)