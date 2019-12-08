#!/usr/bin/env python3
# coding: utf-8

# LGB: Light Gradient Boosted Machine
# Find best LGB hyper-parameters (HyperOpt)


#######################################################################


# Reading files
import toml

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
## Model
import lightgbm as lgb
# Hyper-parameter optimizers
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
# Cross Validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

# Other
import time
import requests


#######################################################################


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
    
# If you want to sort the data, run the cell below
#data.sort(axis=1)

# Convert to Pandas DataFrame
data = pd.DataFrame(data)
data['target'] = target

# Split data into X and y
X = data.drop('target', axis=1)
y = data['target']


#######################################################################


# Define the hyper-parameters space and other parameters
space = {
    'num_leaves': 2 + hp.randint('num_leaves', 150),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'min_data_in_leaf': hp.randint('min_data_in_leaf', 50),
    'alpha': hp.uniform('alpha', 0, 1),
    'lambda': hp.randint('lambda', 100)
}

tpe_algorithm = tpe.suggest
bayes_trials = Trials()
MAX_EVALS = 500
N_FOLDS = 5

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, lgb_train, nfold = n_folds, num_boost_round = 10000, early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    # Extract the best score
    best_score = max(cv_results['auc-mean'])
    # Loss must be minimized
    loss = 1 - best_score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# Run the optimization
lgb_train = lgb.Dataset(X, y)

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

# Print result
# On screen
print('Best result:', best)
# On file
with open('ris/HyperOpt_out.md', mode='a') as f:
    print('# ' + time.ctime(), file=f)
    print('', file=f)
    print('### HyperOpt best result:', file=f)
    print('', file=f)
    print('```python', file=f)
    print(best, file=f)
    print('```', file=f)
    print('', file=f)


# Compute score

# k-fold parameters
n_splits = 5
n_repeats = 6

# Stratified k-fold
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
scores = np.array([])

# Make k-fold CV
i = 1
new_X, new_y = np.array(X.to_numpy()), np.array(y.to_numpy())
for train_index, test_index in rskf.split(new_X, new_y):
    print(i, 'su', n_splits*n_repeats, '\r', end='')
    # Split
    X_train, X_test = new_X[train_index], new_X[test_index]
    y_train, y_test = new_y[train_index], new_y[test_index]
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # Train
    evals_result = {} 
    gbm = lgb.train(best,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_train, lgb_test],
                    evals_result=evals_result,
                    verbose_eval=False)
    # Test
    ris = gbm.predict(X_test)
    # Change into discrete values (yes or no)
    ris[ris>=0.5] = 1
    ris[ris<0.5] = 0
    # Score
    scores = accuracy_score(y_test, ris)
    i += 1

# Print final score
# On screen
print('Score:', scores.mean(), '+-', scores.std())
# On file
with open('ris/HyperOpt_out.md', mode='a') as f:
    print('### Score:', file=f)
    print('', file=f)
    print('```python', file=f)
    print('Score:', scores.mean(), '+-', scores.std(), file=f)
    print('```', file=f)
    print('', file=f)
    print('', file=f)


#######################################################################


# Send telegram message
telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] LGB HyperOpt terminated.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)