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
# Ensemble
from sklearn.ensemble import BaggingClassifier
# Splitter Classes
from sklearn.model_selection import RepeatedStratifiedKFold

# Other
import requests
import threading


#######################################################################


# Multithreading
maxthreads = 4
sema = threading.Semaphore(value=maxthreads)
c = threading.Condition()

# k-fold parameters
n_splits = 5
n_repeats = 6

# Bagging Classifier parameters
n_estimators = 4
max_samples = 0.95

# Reset out file
with open('ris/OUT-score_alglorithms.txt', mode='w') as f:
    pass

telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#######################################################################
# NO MULTI GLITCH
#######################################################################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start no multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
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


###################
# Cross validation


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('k-fold validation.', file=f)

# Multithread function
def thread_function(train_index, test_index):
    # Acquire a semaphore slot
    sema.acquire()
    # Set global variables
    global data
    global target
    global scores
    global best_kernel
    global best_gamma
    global best_C
    # Load training and testing data
    c.acquire()
    clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    c.notify_all()
    c.release()
    # Fit the model
    clf.fit(X_train, y_train)
    train_score = clf.score(X_test, y_test)
    # Save the score
    c.acquire()
    scores = np.append(scores, train_score)
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()

# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#####################
# Data augmentation #
#####################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Start data augmentation part.', file=f)


#################
# Simple training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('k-fold validation.', file=f)

# Multithread function
def thread_function(train_index, test_index):
    # Acquire a semaphore slot
    sema.acquire()
    # Set global variables
    global data
    global target
    global scores
    global scores_aug
    global best_kernel
    global best_gamma
    global best_C
    # Load training and testing data
    c.acquire()
    clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    c.notify_all()
    c.release()
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
    # Fit the model
    clf.fit(X_train_aug, y_train_aug)
    train_score = clf.score(X_test, y_test)
    train_score_aug = clf.score(X_test_aug, y_test_aug)
    # Save the score
    c.acquire()
    scores = np.append(scores, train_score)
    scores_aug = np.append(scores_aug, train_score_aug)
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
scores_aug = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()

# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)
    print('Average score (augmented):', scores_aug.mean(), '+-', scores_aug.std() / np.sqrt(n_splits), file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#############################
# Bagging Classifier training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Bagging Classifier k-fold validation.', file=f)

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
scores_aug = np.array([])
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    # Initialize classifier
    clf = BaggingClassifier(SVC(kernel=best_kernel, gamma=best_gamma, C=best_C), n_estimators=n_estimators, max_samples=max_samples, n_jobs=-1)
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
    # Fit the model
    clf.fit(X_train_aug, y_train_aug)
    train_score = clf.score(X_test, y_test)
    train_score_aug = clf.score(X_test_aug, y_test_aug)
    # Save the score
    scores = np.append(scores, train_score)
    scores_aug = np.append(scores_aug, train_score_aug)
    
# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)
    print('Average score (augmented):', scores_aug.mean(), '+-', scores_aug.std() / np.sqrt(n_splits), file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('End data augmentation part.', file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End no multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('End no multi glitch part.', file=f)


#######################################################################
# YES MULTI GLITCH
#######################################################################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start yes multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('\nStart yes multi glitch part.', file=f)

# Load data
with pd.HDFStore('../../classification/ris/OUT-classified-merged.h5', mode='r') as in_data:
    data = np.concatenate((data, in_data['MULTI_GLITCH'].to_numpy()))
    target = np.concatenate((target, np.ones(len(in_data['MULTI_GLITCH'].to_numpy()))))
                
# Best training parameters
best_kernel = 'rbf'
best_gamma = 0.0151
best_C = 1.45


###################
# Cross validation


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('k-fold validation.', file=f)

# Multithread function
def thread_function(train_index, test_index):
    # Acquire a semaphore slot
    sema.acquire()
    # Set global variables
    global data
    global target
    global scores
    global best_kernel
    global best_gamma
    global best_C
    # Load training and testing data
    c.acquire()
    clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    c.notify_all()
    c.release()
    # Fit the model
    clf.fit(X_train, y_train)
    train_score = clf.score(X_test, y_test)
    # Save the score
    c.acquire()
    scores = np.append(scores, train_score)
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
    
# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#####################
# Data augmentation #
#####################


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Start data augmentation part.', file=f)


#################
# Simple training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('k-fold validation.', file=f)

# Multithread function
def thread_function(train_index, test_index):
    # Acquire a semaphore slot
    sema.acquire()
    # Set global variables
    global data
    global target
    global scores
    global scores_aug
    global best_kernel
    global best_gamma
    global best_C
    # Load training and testing data
    c.acquire()
    clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    c.notify_all()
    c.release()
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
    # Fit the model
    clf.fit(X_train_aug, y_train_aug)
    train_score = clf.score(X_test, y_test)
    train_score_aug = clf.score(X_test_aug, y_test_aug)
    # Save the score
    c.acquire()
    scores = np.append(scores, train_score)
    scores_aug = np.append(scores_aug, train_score_aug)
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
scores_aug = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
    
# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)
    print('Average score (augmented):', scores_aug.mean(), '+-', scores_aug.std() / np.sqrt(n_splits), file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


#############################
# Bagging Classifier training


params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Bagging Classifier k-fold validation.', file=f)

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])
scores_aug = np.array([])
# Make k-fold CV
for train_index, test_index in rskf.split(data, target):
    # Initialize classifier
    clf = BaggingClassifier(SVC(kernel=best_kernel, gamma=best_gamma, C=best_C), n_estimators=n_estimators, max_samples=max_samples, n_jobs=-1)
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
    # Fit the model
    clf.fit(X_train_aug, y_train_aug)
    train_score = clf.score(X_test, y_test)
    train_score_aug = clf.score(X_test_aug, y_test_aug)
    # Save the score
    scores = np.append(scores, train_score)
    scores_aug = np.append(scores_aug, train_score_aug)
    
# Print final score
with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('Average score:', scores.mean(), '+-', scores.std() / np.sqrt(n_splits), file=f)
    print('Average score (augmented):', scores_aug.mean(), '+-', scores_aug.std() / np.sqrt(n_splits), file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('End data augmentation part.', file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End yes multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)

with open('ris/OUT-score_alglorithms.txt', mode='a') as f:
    print('End yes multi glitch part.', file=f)

params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)