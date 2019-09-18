#!/usr/bin/env python3
# coding: utf-8


# MERGE CLASSIFIED DATA
# Merge classified data into three DataFrames: `GLITCH`, `MULTI_GLITCH`
# and `NO_GLITCH`. Columns contain the data, rows contain the samples.


import numpy as np
import pandas as pd

print('START MERGING...')
with pd.HDFStore('ris/OUT-classified.h5', mode='r') as data:
    with pd.HDFStore('ris/OUT-classified-merged.h5', mode='w') as merged_data:
        for group_name in ['/GLITCH', '/MULTI_GLITCH', '/NO_GLITCH']:
            print(group_name)
            group_keys = list(*data.walk(group_name))[2]
            group_keys = np.array(list(map(int, group_keys)))
            group_keys.sort()
            for key in group_keys:
                print(str(key) + '\r', end='')
                df = data[group_name + '/' + str(key)]
                df.columns = [key]
                df = df.transpose()
                df.columns = range(1, 101)
                merged_data.append(group_name, df)
            print()
print('FINISHED.')