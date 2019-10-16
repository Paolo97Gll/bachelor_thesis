#!/usr/bin/env python3
# coding: utf-8

# KNC: K-Neighbors Classifier
# Testing time KNC alglorithms with simple data and data augmentation.


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
with open('ris/OUT-time_predict_alglorithms.txt', mode='w') as f:
    pass

telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Loading data...'}
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

# Sorted data
data_s = np.sort(data, axis=1)

# # Augmented data
# data_aug, target_aug = data, target
# data_aug = np.concatenate((data_aug, -data))
# target_aug = np.concatenate((target_aug, target))
# for j in range(1,100):
#     data_aug = np.concatenate((data_aug, np.roll(data, j, axis=1)))
#     data_aug = np.concatenate((data_aug, -np.roll(data, j, axis=1)))
#     target_aug = np.concatenate((target_aug, target))
#     target_aug = np.concatenate((target_aug, target))

# # Testing data
# n_test_data = 4000000
# test_data = np.random.rand(n_test_data, 100) * 3
# for i in test_data[::2]:
#     if random.getrandbits(1) == 1:
#         i[np.random.randint(100)] += np.random.randint(7,20)
#     if random.getrandbits(1) == 1:
#         i[np.random.randint(100)] += np.random.randint(7,20)
#     if random.getrandbits(1) == 1:
#         i[np.random.randint(100)] += np.random.randint(7,20)
#     if random.getrandbits(1) == 1:
#         i[np.random.randint(100)] += np.random.randint(7,50)
#     if random.getrandbits(1) == 1:
#         i[np.random.randint(100)] += np.random.randint(7,100)
#     if random.getrandbits(1) == 1:
#         i = -i

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Data loaded.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


############
# Standard #
############


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start standard model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_predict_alglorithms.txt', mode='a') as f:
    print('# Standard model.', file=f)

# Best training parameters
best_weights = 'uniform'
best_p = 8
best_n_neighbors = 1
best_leaf_size = 10
best_algorithm = 'ball_tree'

# Classifier
clf = KNeighborsClassifier(n_neighbors=best_n_neighbors,
                           weights=best_weights,
                           algorithm=best_algorithm,
                           leaf_size=best_leaf_size,
                           p=best_p,
                           n_jobs=-1)

# Train
t_b = time.time()
clf.fit(data, target)
t_e = time.time()
with open('ris/OUT-time_predict_alglorithms.txt', mode='a') as f:
    print('Training time (s):', t_e - t_b, file=f)

# Test
predict_time = 0.
for i in range(2000):
    t_b = time.time()
    clf.predict_proba(data)
    t_e = time.time()
    predict_time += t_e - t_b
with open('ris/OUT-time_predict_alglorithms.txt', mode='a') as f:
    print('Prediction time (s):', predict_time, file=f)


###############
# Sorted data #
###############


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start sorted data model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-time_predict_alglorithms.txt', mode='a') as f:
    print('\n# Sorted data model.', file=f)

# Best training parameters
best_weights = 'uniform'
best_p = 3
best_n_neighbors = 1
best_leaf_size = 10
best_algorithm = 'ball_tree'

# Classifier
clf = KNeighborsClassifier(n_neighbors=best_n_neighbors,
                           weights=best_weights,
                           algorithm=best_algorithm,
                           leaf_size=best_leaf_size,
                           p=best_p,
                           n_jobs=-1)

# Train
t_b = time.time()
clf.fit(data_s, target)
t_e = time.time()
with open('ris/OUT-time_predict_alglorithms.txt', mode='a') as f:
    print('Training time (s):', t_e - t_b, file=f)

# Test
predict_time = 0.
for i in range(2000):
    t_b = time.time()
    clf.predict_proba(data_s)
    t_e = time.time()
    predict_time += t_e - t_b
with open('ris/OUT-time_predict_alglorithms.txt', mode='a') as f:
    print('Prediction time (s):', predict_time, file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End sorted data model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#######
# END #
#######


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)
