# SVC - No multi glitch

## Standard training

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

## Data augmentation training

### Normal SVC

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

### Bagging Classifier with SVC

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

## Sorted training

_Best hyper-parameters:_

```
best_kernel = 'rbf'
best_gamma = 0.0145
best_C = 0.8
```

_Score:_

```
0.9989511376976302 +- 0.0016290507789316132
```