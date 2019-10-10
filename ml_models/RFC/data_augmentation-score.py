#!/usr/bin/env python3
# coding: utf-8

# RFC: Random Forest Classifier
# Score RFC alglorithms with simple data and data augmentation.


#######################################################################


# Reading files
import toml

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
# Model
from sklearn.ensemble import RandomForestClassifier
# Splitter Classes
from sklearn.model_selection import RepeatedStratifiedKFold

# Other
import time
import requests


#######################################################################


# k-fold parameters
n_splits = 5
n_repeats = 6

# Reset out file
with open('ris/OUT-score_alglorithms.txt', mode='w') as f:
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
best_bootstrap = False
best_max_depth = None
best_max_features = 'sqrt'
best_min_samples_leaf = 1
best_min_samples_split = 3
best_n_estimators = 400


##########
# Simple #
##########


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start simple model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('# Simple model.', file=f)


####################
# k-fold validation


# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    
    # Initialize model
    clf = RandomForestClassifier(bootstrap=best_bootstrap,
                                 max_depth=best_max_depth,
                                 max_features=best_max_features,
                                 min_samples_leaf=best_min_samples_leaf,
                                 min_samples_split=best_min_samples_split,
                                 n_estimators=best_n_estimators,
                                 n_jobs=-1)
    # Split data
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    # Fit and score the model
    clf.fit(X_train, y_train)
    train_score = clf.score(X_test, y_test)
    scores = np.append(scores, train_score)

# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)


################
# training time


clf = RandomForestClassifier(bootstrap=best_bootstrap,
                             max_depth=best_max_depth,
                             max_features=best_max_features,
                             min_samples_leaf=best_min_samples_leaf,
                             min_samples_split=best_min_samples_split,
                             n_estimators=best_n_estimators,
                             n_jobs=-1)
t_b = time.time()
clf.fit(data,target)
t_e = time.time()

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)


#####################
# Data augmentation #
#####################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('\n# Data augmentation model.', file=f)


####################
# k-fold validation


# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
scores_aug = np.array([])
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    
    # Initialize model
    clf = RandomForestClassifier(bootstrap=best_bootstrap,
                                 max_depth=best_max_depth,
                                 max_features=best_max_features,
                                 min_samples_leaf=best_min_samples_leaf,
                                 min_samples_split=best_min_samples_split,
                                 n_estimators=best_n_estimators,
                                 n_jobs=-1)
    
    # Split data
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    # Apply data augmentation on training data
    X_train_aug, y_train_aug = X_train, y_train
    X_train_aug = np.concatenate((X_train_aug, -X_train))
    y_train_aug = np.concatenate((y_train_aug, y_train))
    for j in range(1,100):
        X_train_aug = np.concatenate((X_train_aug, np.roll(X_train, j, axis=1)))
        X_train_aug = np.concatenate((X_train_aug, -np.roll(X_train, j, axis=1)))
        y_train_aug = np.concatenate((y_train_aug, y_train))
        y_train_aug = np.concatenate((y_train_aug, y_train))
    # Apply data augmentation on testing data
    X_test_aug, y_test_aug = X_test, y_test
    X_test_aug = np.concatenate((X_test_aug, -X_test))
    y_test_aug = np.concatenate((y_test_aug, y_test))
    for j in range(1,100):
        X_test_aug = np.concatenate((X_test_aug, np.roll(X_test, j, axis=1)))
        X_test_aug = np.concatenate((X_test_aug, -np.roll(X_test, j, axis=1)))
        y_test_aug = np.concatenate((y_test_aug, y_test))
        y_test_aug = np.concatenate((y_test_aug, y_test))
    
    # Fit and score the model
    clf.fit(X_train_aug, y_train_aug)
    train_score = clf.score(X_test, y_test)
    train_score_aug = clf.score(X_test_aug, y_test_aug)
    scores = np.append(scores, train_score)
    scores_aug = np.append(scores_aug, train_score_aug)

# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)
    print('Average score (augmented):', scores_aug.mean(), '+-', scores_aug.std() / np.sqrt(n_splits), file=f)


################
# training time


clf = RandomForestClassifier(bootstrap=best_bootstrap,
                             max_depth=best_max_depth,
                             max_features=best_max_features,
                             min_samples_leaf=best_min_samples_leaf,
                             min_samples_split=best_min_samples_split,
                             n_estimators=best_n_estimators,
                             n_jobs=-1)
data_aug, target_aug = data, target
data_aug = np.concatenate((data_aug, -data))
target_aug = np.concatenate((target_aug, target))
for j in range(1,100):
    data_aug = np.concatenate((data_aug, np.roll(data, j, axis=1)))
    data_aug = np.concatenate((data_aug, -np.roll(data, j, axis=1)))
    target_aug = np.concatenate((target_aug, target))
    target_aug = np.concatenate((target_aug, target))
t_b = time.time()
clf.fit(data_aug, target_aug)
t_e = time.time()

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)


##########
# Sorted #
##########


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start sorted model.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('\n# Sorted model.', file=f)


####################
# k-fold validation


best_bootstrap = False
best_max_depth = 40
best_max_features = 'sqrt'
best_min_samples_leaf = 1
best_min_samples_split = 2
best_n_estimators = 200

data.sort(axis=1)

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    
    # Initialize model
    clf = RandomForestClassifier(bootstrap=best_bootstrap,
                                 max_depth=best_max_depth,
                                 max_features=best_max_features,
                                 min_samples_leaf=best_min_samples_leaf,
                                 min_samples_split=best_min_samples_split,
                                 n_estimators=best_n_estimators,
                                 n_jobs=-1)
    # Split data
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    # Fit and score the model
    clf.fit(X_train, y_train)
    train_score = clf.score(X_test, y_test)
    scores = np.append(scores, train_score)

# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)


################
# training time


clf = RandomForestClassifier(bootstrap=best_bootstrap,
                             max_depth=best_max_depth,
                             max_features=best_max_features,
                             min_samples_leaf=best_min_samples_leaf,
                             min_samples_split=best_min_samples_split,
                             n_estimators=best_n_estimators,
                             n_jobs=-1)
t_b = time.time()
clf.fit(data,target)
t_e = time.time()

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Time (s):', t_e - t_b, file=f)


######################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)