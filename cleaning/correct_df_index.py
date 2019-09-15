#!/usr/bin/env python3
# coding: utf-8


# CORRECT DATAFRAME INDEX
# Correct pandas DataFrame index of the "ris/OUT-cleaned.h5" file:
# the new index is the 'time_cleaned' column.


import os
if not os.path.exists('ris') or not os.path.exists('ris/OUT-cleaned.h5'):
    exit(0)


import pandas as pd
import h5py

with pd.HDFStore('ris/OUT-cleaned.h5', mode='r') as data:
    with pd.HDFStore('ris/OUT-cleaned-new.h5') as new_data:
        for df_name in data:
            print('DataFrame:', df_name)
            df = data[df_name]
            df.set_index(['time_cleaned'], inplace=True)
            new_data.put(df_name, df)
            
with h5py.File('ris/OUT-cleaned.h5', mode='r') as data:
    with h5py.File('ris/OUT-cleaned-new.h5') as new_data:
        new_data.attrs['TITLE'] = data.attrs['TITLE']
        new_data.attrs['VERSION'] = data.attrs['VERSION']
        
os.remove('ris/OUT-cleaned.h5')
os.rename('ris/OUT-cleaned-new.h5', 'ris/OUT-cleaned.h5')

print('FINISHED.')