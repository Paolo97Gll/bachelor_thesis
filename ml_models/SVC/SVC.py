#!/usr/bin/env python3
# coding: utf-8

# # SVC: C-Support Vector Classification
#
# A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side. (from [here](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72))


# Reading files
import h5py
import toml

# Scientific computing
import numpy as np
import pandas as pd
from scipy import interp

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#sns.set_context('paper')

# Machine Learning
# Model
from sklearn.svm import SVC
# Ensemble
from sklearn.ensemble import BaggingClassifier
# Splitter Classes
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
# Splitter Functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
# Hyper-parameter optimizers
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Model validation
from sklearn.model_selection import learning_curve
# Training metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Other
import os
import time
import requests
import threading

# Globally redirect print statements to stdout
import sys
sys.stdout = open('SVC_py_out.txt', mode='w')

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] ####################'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# # NO MULTI GLITCH

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start no multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# ## Preparation
#
# Load data and target from `classification/ris/OUT-classified-merged.h5` and load into numpy arrays.
#
# **Label `0` = NO GLITCH**
#
# **Label `1` = GLITCH**

# In[ ]:


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


# ## Best training
#
# Initialize best hyper-parameters founded.

# In[ ]:


best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8


# ### Cross validation
#
# Use k-fold to make a cross validation of the model.

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


maxthreads = 4
sema = threading.Semaphore(value=maxthreads)
c = threading.Condition()

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
    # Save the score
    c.acquire()
    scores = np.append(scores, clf.score(X_test, y_test))
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()


## K-FOLD

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rkf.split(data, target):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
# Print final score
print('Average score (k-fold):', scores.mean(), '+-', scores.std())


## STRATIFIED K-FOLD

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)
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
print('Average score (Stratified k-fold):', scores.mean(), '+-', scores.std())


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# ## Data augmentation
#
# Data augmentation is a strategy that increase the diversity of data available for training models, without actually collecting new data. The data augmentation techniques used in this situation are vertical flipping and translation.

# ### Preparation

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


data_aug, target_aug = data, target
data_aug = np.concatenate((data_aug, -data))
target_aug = np.concatenate((target_aug, target))

for i in range(1,100):
    data_aug = np.concatenate((data_aug, np.roll(data, i, axis=1)))
    data_aug = np.concatenate((data_aug, -np.roll(data, i, axis=1)))
    target_aug = np.concatenate((target_aug, target))
    target_aug = np.concatenate((target_aug, target))


# ### Simple training

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


maxthreads = 4
sema = threading.Semaphore(value=maxthreads)
c = threading.Condition()

def thread_function(train_index, test_index):
    # Acquire a semaphore slot
    sema.acquire()
    # Set global variables
    global data_aug
    global target_aug
    global scores
    global best_kernel
    global best_gamma
    global best_C
    # Load training and testing data
    c.acquire()
    clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
    X_train, X_test = data_aug[train_index], data_aug[test_index]
    y_train, y_test = target_aug[train_index], target_aug[test_index]
    c.notify_all()
    c.release()
    # Fit the model
    clf.fit(X_train, y_train)
    # Save the score
    c.acquire()
    scores = np.append(scores, clf.score(X_test, y_test))
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()


## K-FOLD

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rkf.split(data_aug, target_aug):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
# Print final score
print('Average score (k-fold):', scores.mean(), '+-', scores.std())


## STRATIFIED K-FOLD

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rskf.split(data_aug, target_aug):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
# Print final score
print('Average score (Stratified k-fold):', scores.mean(), '+-', scores.std())


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# ### Bagging Classifier training
#
# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


n_estimators = 10
clf = BaggingClassifier(SVC(kernel=best_kernel, gamma=best_gamma, C=best_C), n_estimators=n_estimators, max_samples=1./n_estimators, n_jobs=-1)


## K-FOLD

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
scores_rkf = np.array([])
# Make k-fold CV
for train_index, test_index in rkf.split(data_aug, target_aug):
    X_train, X_test = data_aug[train_index], data_aug[test_index]
    y_train, y_test = target_aug[train_index], target_aug[test_index]
    clf.fit(X_train, y_train)
    scores_rkf = np.append(scores_rkf, clf.score(X_test, y_test))
# Print final score
print('Average score (k-fold):', scores_rkf.mean(), '+-', scores_rkf.std())


## STRATIFIED K-FOLD

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)
scores_rskf = np.array([])
# Make k-fold CV
for train_index, test_index in rskf.split(data_aug, target_aug):
    X_train, X_test = data_aug[train_index], data_aug[test_index]
    y_train, y_test = target_aug[train_index], target_aug[test_index]
    clf.fit(X_train, y_train)
    scores_rskf = np.append(scores_rskf, clf.score(X_test, y_test))
# Print final score
print('Average score (Stratified k-fold):', scores_rskf.mean(), '+-', scores_rskf.std())


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End no multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# # YES MULTI GLITCH

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start yes multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# ## Preparation
#
# Load data and target from `classification/ris/OUT-classified-merged.h5` and load into numpy arrays.
#
# **Label `0` = NO GLITCH**
#
# **Label `1` = GLITCH and MULTI GLITCH**

# In[ ]:


with pd.HDFStore('../../classification/ris/OUT-classified-merged.h5', mode='r') as in_data:
    data = np.concatenate((data, in_data['MULTI_GLITCH'].to_numpy()))
    target = np.concatenate((target, np.ones(len(in_data['MULTI_GLITCH'].to_numpy()))))


# ## Best training
#
# Initialize best hyper-parameters founded.

# In[ ]:


best_kernel = 'rbf'
best_gamma = 0.0151
best_C = 1.45


# ### Cross validation
#
# Use k-fold to make a cross validation of the model.

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


maxthreads = 4
sema = threading.Semaphore(value=maxthreads)
c = threading.Condition()

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
    # Save the score
    c.acquire()
    scores = np.append(scores, clf.score(X_test, y_test))
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()


## K-FOLD

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rkf.split(data, target):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
# Print final score
print('Average score (k-fold):', scores.mean(), '+-', scores.std())


## STRATIFIED K-FOLD

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)
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
print('Average score (Stratified k-fold):', scores.mean(), '+-', scores.std())


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# ## Data augmentation
#
# Data augmentation is a strategy that increase the diversity of data available for training models, without actually collecting new data. The data augmentation techniques used in this situation are vertical flipping and translation.

# ### Preparation

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


data_aug, target_aug = data, target
data_aug = np.concatenate((data_aug, -data))
target_aug = np.concatenate((target_aug, target))

for i in range(1,100):
    data_aug = np.concatenate((data_aug, np.roll(data, i, axis=1)))
    data_aug = np.concatenate((data_aug, -np.roll(data, i, axis=1)))
    target_aug = np.concatenate((target_aug, target))
    target_aug = np.concatenate((target_aug, target))


# ### Simple training

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


maxthreads = 4
sema = threading.Semaphore(value=maxthreads)
c = threading.Condition()

def thread_function(train_index, test_index):
    # Acquire a semaphore slot
    sema.acquire()
    # Set global variables
    global data_aug
    global target_aug
    global scores
    global best_kernel
    global best_gamma
    global best_C
    # Load training and testing data
    c.acquire()
    clf = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
    X_train, X_test = data_aug[train_index], data_aug[test_index]
    y_train, y_test = target_aug[train_index], target_aug[test_index]
    c.notify_all()
    c.release()
    # Fit the model
    clf.fit(X_train, y_train)
    # Save the score
    c.acquire()
    scores = np.append(scores, clf.score(X_test, y_test))
    c.notify_all()
    c.release()
    # Release the semaphore slot
    sema.release()


## K-FOLD

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rkf.split(data_aug, target_aug):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
# Print final score
print('Average score (k-fold):', scores.mean(), '+-', scores.std())


## STRATIFIED K-FOLD

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)
scores = np.array([])
threads = []
# Make k-fold CV
for train_index, test_index in rskf.split(data_aug, target_aug):
    thread = threading.Thread(target=thread_function, args=(train_index, test_index))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
# Print final score
print('Average score (Stratified k-fold):', scores.mean(), '+-', scores.std())


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# ### Bagging Classifier training
#
# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.

# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] Start Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


n_estimators = 10
clf = BaggingClassifier(SVC(kernel=best_kernel, gamma=best_gamma, C=best_C), n_estimators=n_estimators, max_samples=1./n_estimators, n_jobs=-1)


## K-FOLD

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
scores_rkf = np.array([])
# Make k-fold CV
for train_index, test_index in rkf.split(data_aug, target_aug):
    X_train, X_test = data_aug[train_index], data_aug[test_index]
    y_train, y_test = target_aug[train_index], target_aug[test_index]
    clf.fit(X_train, y_train)
    scores_rkf = np.append(scores_rkf, clf.score(X_test, y_test))
# Print final score
print('Average score (k-fold):', scores_rkf.mean(), '+-', scores_rkf.std())


## STRATIFIED K-FOLD

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)
scores_rskf = np.array([])
# Make k-fold CV
for train_index, test_index in rskf.split(data_aug, target_aug):
    X_train, X_test = data_aug[train_index], data_aug[test_index]
    y_train, y_test = target_aug[train_index], target_aug[test_index]
    clf.fit(X_train, y_train)
    scores_rskf = np.append(scores_rskf, clf.score(X_test, y_test))
# Print final score
print('Average score (Stratified k-fold):', scores_rskf.mean(), '+-', scores_rskf.std())


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End Bagging Classifier k-fold validation.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End data augmentation part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)


# In[ ]:


telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] End yes multi glitch part.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)
