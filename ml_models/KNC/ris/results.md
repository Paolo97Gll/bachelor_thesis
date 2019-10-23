# Without probability

## Cross validation

_Cross validation method:_

```
RepeatedStratifiedKFold(n_splits=5, n_repeats=6)
```

So, `training_data` are 0.8 of `data` and `testing_data` are 0.2 of `data`.

While computing the training time, `training_data = data`.

## Standard training

_Data info:_

```
data.shape = (2000, 100)
```

_Best hyper-parameters:_

```
FILL
```

_Score:_

```
FILL
```

_Training time:_

```
0.106 s
```

## Data augmentation training

_Data info:_

```
data.shape = (400000, 100)
```

_Best hyper-parameters:_

```
FILL
```

_Score:_

- Normal test dataset:

    ```
    FILL
    ```

- Augmented test dataset:

    ```
    FILL
    ```

_Training time:_

```
45.476 s
```

## Sorted training

_Data info:_

```
data.shape = (2000, 100)
```

_Best hyper-parameters:_

```
FILL
```

_Score:_

```
FILL
```

_Training time:_

```
0.097 s
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

Same as above.

_Testing time:_

```
4 h  0 m  39.048 s
```

## Sorted model

_Training time:_

Same as above.

_Testing time:_

```
1 h  23 m  32.290 s
```
