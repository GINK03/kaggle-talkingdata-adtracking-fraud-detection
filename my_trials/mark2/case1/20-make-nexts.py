import pandas as pd
import gc

import time
import numpy as np

import itertools
import os
import gc
import concurrent.futures

print('loading dataset')
dft = pd.read_csv('../../input/train.csv', parse_dates=['click_time'])
dfT = pd.read_csv('../../input/test.csv', parse_dates=['click_time'])
dft = dft.append(dfT)
dft = dft[ ['click_time', 'ip', 'os', 'app', 'device'] ]
gc.collect()
dft.reset_index()
print(len(dft))
print('finish loading')

def pmap(arg):
  factN = arg
  #if os.path.exists(save_key):
  #  return
  print('slicing now.')
  df = dft[ ['click_time', 'ip', 'os', 'app', 'device'] ]
  print('df len', len(df))
  gc.collect()
  print('slicing was done.')
  
  df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)

  print('calculate delta time now of next.')
  save_key = 'var/' + '_'.join(factN) + '_nextclick.csv'
  key = '_'.join(factN)
  
  df[key] = (df.groupby(factN).click_time.shift(-1) - df.click_time).astype(np.float32)
  series = df[ key ] 
  gc.collect()
  print('calculate delta time of next was done.')
  print('dump to csv now.')
  series.to_csv(save_key, index=False, header=None)
  print('dump to csv was done.')
  
  print('calculate delta time now of nextnext.')
  save_key = 'var/' + '_'.join(factN) + '_nextnextclick.csv'
  key = '_'.join(factN)
  df[key] = (df.groupby(factN).click_time.shift(-2) - df.click_time).astype(np.float32)
  series = df[ key ] 
  gc.collect()
  print('calculate delta time of nextnext was done.')
  print('dump to csv now.')
  series.to_csv(save_key, index=False, header=None)
  print('dump to csv was done.')


  print('calculate delta time now of prev.')
  save_key = 'var/' + '_'.join(factN) + '_prevclick.csv'
  key = '_'.join(factN)
  df[key] = (df.click_time - df.groupby(factN).click_time.shift(+1)).astype(np.float32)
  series = df[ key ] 
  gc.collect()
  print('calculate delta time was done.')
  print('dump to csv now.')
  series.to_csv(save_key, index=False, header=None)
  print('dump to csv was done.')

  del df; del series; gc.collect()

args = []
for fact in reversed( [ itertools.combinations( ['ip', 'os', 'app', 'device'], i )  for i in range(2,5) ] ): 
  for factN in fact:
    args.append(factN)
with concurrent.futures.ProcessPoolExecutor(max_workers=18) as exe:
  exe.map(pmap, args)
