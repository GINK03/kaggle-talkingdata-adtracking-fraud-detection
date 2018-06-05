import pandas as pd

import sys

if '--init' in sys.argv:
  df = pd.read_csv('../input/train.csv')
  df_test = pd.read_csv('../input/train.csv')
  df.append( df_test )
  
  df['factrize'] = df['ip'].astype(str) + "_" + df['channel'].astype(str)
  buff = []
  for index, factrize in enumerate( df['factrize'].unique().tolist() ):
    obj = {'ip_chl_ind':index, 'factrize': factrize}
    buff.append( obj )
  key_df = pd.DataFrame( buff )
  key_df.to_pickle( 'files/key_df.pkl' )
if '--merge' in sys.argv:
  key_df = pd.read_pickle('./files/key_df.pkl')
  
  print('train merge')
  df = pd.read_pickle('./files/train_df.pkl')
  df['factrize'] = df['ip'].astype(str) + "_" + df['channel'].astype(str)
  df = df.merge(key_df, on=['factrize'], how='left')
  df.to_pickle('./files/train_df.pkl')

  print('test merge')
  df = pd.read_pickle('./files/test_df.pkl')
  df['factrize'] = df['ip'].astype(str) + "_" + df['channel'].astype(str)
  df = df.merge(key_df, on=['factrize'], how='left')
  df.to_pickle('./files/test_df.pkl')

  print('val merge')
  df = pd.read_pickle('./files/val_df.pkl')
  df['factrize'] = df['ip'].astype(str) + "_" + df['channel'].astype(str)
  df = df.merge(key_df, on=['factrize'], how='left')
  df.to_pickle('./files/val_df.pkl')
#df.info()
#print(df.head())
