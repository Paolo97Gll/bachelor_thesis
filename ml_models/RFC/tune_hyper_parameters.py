#!/usr/bin/env python3
# coding: utf-8

# RFC: Random Forest Classifier
# Find best RFC hyper-parameters


#######################################################################


# Reading files
import toml

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
## Model
from sklearn.ensemble import RandomForestClassifier
# Hyper-parameter optimizers
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

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


#######################################################################


# Number of trees in random forest
n_estimators = [int(x) for x in np.arange(100, 1000, 100)]
# Number of features to consider at every split
max_features = [None, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.arange(10, 100, 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = range(2,11)
# Minimum number of samples required at each leaf node
min_samples_leaf = range(1,5)
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Grid
parameters = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

# Start search
GSCV = RandomizedSearchCV(RandomForestClassifier(), parameters, n_iter=1e3, n_jobs=-1, cv=3, iid=False)
print('START')
t_b = time.time()
GSCV.fit(data, target)
t_e = time.time()

# Print best parameters
print('Best parameters set found on development set:', GSCV.best_params_)
print('Score:', GSCV.best_score_)
print('Time (s):', t_e - t_b)

# Print into a file the grid score
with open('ris/RandomizedSearch_mg_out.md', mode='a') as f:
    print('# ' + time.ctime(), file=f)
    print('', file=f)
    print('### RandomizedSearchCV parameters:', file=f)
    print('', file=f)
    print('```python', file=f)
    print(GSCV.get_params, file=f)
    print('```', file=f)
    print('', file=f)
    print('### Best SVC parameters:', file=f)
    print('', file=f)
    print('```python', file=f)
    print(GSCV.best_estimator_, file=f)
    print('```', file=f)
    print('', file=f)
    print('### Best parameters set found on development set:', file=f)
    print('', file=f)
    print('```python', file=f)
    print(GSCV.best_params_, file=f)
    print('```', file=f)
    print('', file=f)
    print('### Grid scores on development set:', file=f)
    print('', file=f)
    print('```', file=f)
    means = GSCV.cv_results_['mean_test_score']
    stds = GSCV.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, GSCV.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r'
              % (mean, std * 2, params), file=f)
    print('```', file=f)
    print('', file=f)
    print('', file=f)

# Send telegram message
telegram_bot_id = toml.load('../telegram_bot_id.toml')
params = {'chat_id': telegram_bot_id['chat_id'], 'text': '[python] RFC randomized search terminated.'}
requests.post('https://api.telegram.org/' + telegram_bot_id['bot_id'] + '/sendMessage', params=params)
