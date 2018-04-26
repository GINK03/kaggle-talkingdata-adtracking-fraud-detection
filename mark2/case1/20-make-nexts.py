import pandas as pd
import gc

import time
import numpy as np

import itertools

print('loading dataset')
dft = pd.read_csv('../../input/train.csv', parse_dates=['click_time'])
dfT = pd.read_csv('../../input/test.csv', parse_dates=['click_time'])
dft = dft.append(dfT)
dft.reset_index()
print(len(dft))
print('finish loading')
#sys.exit()
for fact in [ itertools.combinations( ['ip', 'os', 'app', 'device'], i )  for i in range(2,4) ] : 
  for factN in fact:
    print('slicing now.')
    df = dft[ ['click_time', 'ip', 'os', 'app', 'device'] ]
    print('df len', len(df))
    gc.collect()
    print('slicing was done.')

    print('calculate delta time now.')
    key = '_'.join(factN)
    df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    df[key] = (df.groupby(factN).click_time.shift(-1) - df.click_time).astype(np.float32)
  
    df['control_index'] = df.index
    df = df[['control_index', key]] 
    gc.collect()
    print('calculate delta time was done.')

    print('dump to csv now.')
    save_key = 'var/' + '_'.join(factN) + '_nextclick.csv'
    df.to_csv(save_key, index=False)
    print('dump to csv was done.')
    del df; gc.collect()
