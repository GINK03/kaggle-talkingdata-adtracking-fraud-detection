import pandas as pd

import sys

if '--convert' in sys.argv:
  # min tansの計算
  df = pd.read_csv('../input/train.csv', nrows=100)
  df = df.head(1000)
  trans = pd.to_numeric(pd.to_datetime(df['click_time'])) / (10**12)
  min_trans = trans.min()
  
  print('val convert')
  df = pd.read_pickle('./files/val_df.pkl')
  trans = pd.to_numeric( df['click_time'] ) / (10**12) - min_trans
  df['epochtime'] = trans
  print(df.tail())
  df.to_pickle('./files/val_df.pkl')
  
  print('train convert')
  df = pd.read_pickle('./files/train_df.pkl')
  trans = pd.to_numeric( df['click_time'] ) / (10**12) - min_trans
  df['epochtime'] = trans
  print(df.tail())
  df.to_pickle('./files/train_df.pkl')

  print('test merge')
  df = pd.read_pickle('./files/test_df.pkl')
  trans = pd.to_numeric( df['click_time'] ) / (10**12) - min_trans
  df['epochtime'] = trans
  print(df.tail())
  df.to_pickle('./files/test_df.pkl')

#df.info()
#print(df.head())
