# Without probability

## Cross validation

_Cross validation method:_

```
RepeatedStratifiedKFold(n_splits=5, n_repeats=6)
```

So, `training_data` are 0.8 of `data` and `testing_data` are 0.2 of `data`.

While computing the training time, `training_data = data`.

## No multi glitch

### Standard training

_Data info:_

```
data.shape = (1735, 100)
```

_Best hyper-parameters:_

```
best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8
```

_Score:_

```
0.9785693372600403 +- 0.007206758240810251
```

_Training time:_

```
0.381 s
```

### Data augmentation training

_Data info:_

```
data.shape = (347000, 100)
```

#### Normal SVC

_Best hyper-parameters:_

```
best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8
```

_Score:_

- Normal test dataset:

    ```
    0.9879928336515279 +- 0.0028638819815549246
    ```

- Augmented test dataset:

    ```
    0.9879928336515279 +- 0.0028638819815549246
    ```

_Training time:_

```
1 h  8 m  41.788 s
```

#### Bagging Classifier with SVC

_Best hyper-parameters:_

```
best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8

n_estimators = 4
max_samples = 0.95
```

_Score:_

- Normal test dataset:

    ```
    0.9887607686220358 +- 0.0021904060021883673
    ```

- Augmented test dataset:

    ```
    0.9887974233667013 +- 0.0021588713860837966
    ```

_Training time:_
```
58 m  11.575 s
```

### Sorted training

_Data info:_

```
data.shape = (1735, 100)
```

_Best hyper-parameters:_

```
best_kernel = 'linear'
best_C = 0.117
```

_Score:_

```
0.9989511376976302 +- 0.0016290507789316132
```

_Training time:_

```
0.019 s
```

## Yes multi glitch

### Standard training

_Data info:_

```
data.shape = (2000, 100)
```

_Best hyper-parameters:_

```
best_kernel = 'rbf'
best_gamma = 0.0151
best_C = 1.45
```

_Score:_

```
0.9805394027462672 +- 0.00627073597526611
```

_Training time:_

```
0.506 s
```

### Data augmentation training

_Data info:_

```
data.shape = (400000, 100)
```

#### Normal SVC

_Best hyper-parameters:_

```
best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8
```

_Score:_

- Normal test dataset:

    ```
    0.9896635150844693 +- 0.0021393523591691794
    ```

- Augmented test dataset:

    ```
    0.9896635150844693 +- 0.0021393523591691794
    ```

_Training time:_

```
1 h  53 m  50.236 s
```

#### Bagging Classifier with SVC

_Best hyper-parameters:_

```
best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8

n_estimators = 4
max_samples = 0.95
```

_Score:_

- Normal test dataset:

    ```
    0.98966560051417 +- 0.0020375628427123636
    ```

- Augmented test dataset:

    ```
    0.9898043073784837 +- 0.001868856310543336
    ```

_Training time:_

```
1 h  21 m  1.844 s
```

### Sorted training

_Data info:_

```
data.shape = (2000, 100)
```

_Best hyper-parameters:_

```
best_kernel = 'linear'
best_C = 0.15
```

_Score:_

```
0.9993199733123334 +- 0.0012401420493798475
```

_Training time:_

```
0.017 s
```

# With probability: computational time

Best hyper-parameters and score are not computed as they are the same as above.

_Data info:_

```
data.shape = (2000, 100)
```

Every model has been tested using `data` and repeating the test 2000 times. So, the testing sample is about 4'000'000 elements. The final time reported is the sum of every single testing time.

## Standard model

_Training time:_

```
2.639 s
```

_Testing time:_

```
12 m  47.225 s
```

## Bagging Classifier model

_Training time:_

```
2.692 s
```

_Testing time:_

```
9 m  57.627 s
```

## Sorted model

_Training time:_

```
0.766 s
```

_Testing time:_

```
3 m  9.198 s
```
