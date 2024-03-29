
import time
import numpy as np
from   sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import pandas as pd
import pickle
import gzip
import os
import sys
import random
import json
import hashlib

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        'qty'           : 'uint16',
        'ip_app_count'  : 'uint16',
        'ip_app_os_count': 'uint16',
        'ip_os_hour_count': 'uint16', 
        'p_os_app_hour_count': 'uint16',
        'dh_f': 'uint16',
        }
if '--prepare' in sys.argv:
    #for window in [ i*500_000 for i in range(100) ]:
    window = 500_0000
    print('load train...')
    train_df = pd.read_csv("inputs/train.csv", skiprows=range(1,134903891-window), nrows=40000000+window, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    print(' load test...')
    test_df = pd.read_csv("inputs/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df  = train_df.append(test_df)

    print('data prep...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day']  = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

    # # of clicks for each ip-day-hour combination
    print('make:qty, group by...["ip", "day", "hour", "channel"] ')
    gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left'); del gp

    print('make:ip_channle, group by...["ip", "channel"] ')
    gp = train_df[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_channel'})
    train_df = train_df.merge(gp, on=['ip'], how='left'); del gp

    print('make:device_channle, group by...["device", "channel"] ')
    gp = train_df[['device','channel']].groupby(by=['device'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'device_channel'})
    train_df = train_df.merge(gp, on=['device'], how='left'); del gp
    
    print('make:device_hour_channle, group by...["device", "hour" , "channel"] ')
    gp = train_df[['device', 'hour', 'channel']].groupby(by=['device', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'device_hour_channel'})
    train_df = train_df.merge(gp, on=['device', 'hour'], how='left'); del gp

    print('make:os_channle, group by...["os", "channel"] ')
    gp = train_df[['os','channel']].groupby(by=['os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'os_channel'})
    train_df = train_df.merge(gp, on=['os'], how='left'); del gp

    print('make:ip_app_chl_mean_hour, group by...["ip", "app", "channel"]')
    gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left'); del gp

    print('make:ip_app_uniq, group by...["ip", "app", "channel"]')
    gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_app_uniq'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left'); del gp

    print('make:ip-day-hour, group by...["ip", "day", "hour"] ')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left'); del gp

    print('make:ip_day_chl_var_hour, group by...["ip", "day", "channel"]')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_hour_var'})
    train_df = train_df.merge(gp, on=['ip','day','channel'], how='left'); del gp

    # --- 
    print('make:ip_app_os_var_hour, group by...["ip", "app", "os"]')
    gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left'); del gp

    # --- 
    print('make:ip_app_channel_var_day, group by...["ip", "app", "channel"]')
    gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left'); del gp

    # # of clicks for each ip-app combination
    print('make:ip_app_channel, group by...["ip", "app", "channel"]')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left'); del gp

    print('make:ip_os_hour_count, group by...["ip", "hour", "os"]')
    gp = train_df[['ip', 'hour', 'os', 'channel']].groupby(by=['ip', 'hour', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_hour_count'})
    train_df = train_df.merge(gp, on=['ip', 'hour', 'os'], how='left'); del gp
    
    print('make:ip_app_os_count, group by...["ip", "app", "os", "channel"]')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left'); del gp
    
    print('make:ip_os_app_hour_count, group by...["ip", "os", "app", "hour"]')
    gp = train_df[['ip', 'os', 'app', 'hour', 'channel']].groupby(by=['ip', 'os', 'app', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_app_hour_count'})
    train_df = train_df.merge(gp, on=['ip', 'os', 'app', 'hour'], how='left'); del gp

    print('make:ip_device_uniq, group by...["ip", "device"]')
    gp = train_df[['ip', 'os', 'app', 'device', 'channel']].groupby(by=['ip', 'device'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_device_uniq'})
    train_df = train_df.merge(gp, on=['ip', 'device' ], how='left'); del gp

    print('make:ip_app_uniq, group by...["ip", "os", "app"]')
    gp = train_df[['ip', 'os', 'app', 'device', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_device_uniq'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left'); del gp

    print('make:ip_os_app_device_uniq, group by...["ip", "os", "app", "device"]')
    gp = train_df[['ip', 'os', 'app', 'device', 'channel']].groupby(by=['ip', 'os', 'app', 'device'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_os_app_device_uniq'})
    train_df = train_df.merge(gp, on=['ip', 'os', 'app', 'device'], how='left'); del gp

    print("vars and data type: ")
    train_df['qty']             = train_df['qty'].astype('uint16')
    train_df['ip_app_count']    = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    # ここを編集した
    test_df  = train_df[len_train:]
    val_df   = train_df[(len_train-2500000):len_train]
    train_df = train_df[:(len_train-2500000)]

    print('train size: ', len(train_df))
    print('valid size: ', len(val_df))
    print('test size : ', len(test_df))

    #train_df.to_pickle( f'files/train_df_{window:012d}.pkl.gz',  'gzip')
    #val_df.to_pickle(   f'files/val_df_{window:012d}.pkl.gz',    'gzip')
    #test_df.to_pickle(  f'files/test_df_{window:012d}.pkl.gz',   'gzip')
    train_df.to_csv( f'files/train_df_{window:012d}.csv.gz', compression='gzip')
    val_df.to_csv(   f'files/val_df_{window:012d}.csv.gz', compression='gzip')
    test_df.to_csv(  f'files/test_df_{window:012d}.csv.gz', compression='gzip')

if '--train' in sys.argv:
  trials = [ i for i in range(100) ]
  random.shuffle(trials)
  for trial in trials:
    window = 500_0000
    base = ['ip','app','device','os','channel','is_attributed','hour','day']
    extend = 'qty,ip_channel,device_channel,device_hour_channel,os_channel,ip_app_channel_mean_hour,ip_app_uniq,ip_tcount,ip_hour_var,ip_app_os_var,ip_app_channel_var_day,ip_app_count,ip_os_hour_count,ip_app_os_count,ip_os_app_hour_count,ip_device_uniq_x,ip_device_uniq_y,ip_os_app_device_uniq'.split(',')

    base.extend( extend )
    try:
      print('load to csv files')
      _mid = ''
      train_df = pd.read_csv(f'files/train_df{_mid}_{window:012d}.csv.gz', dtype=dtypes, usecols=base, compression='gzip')
      val_df = pd.read_csv(f'files/val_df{_mid}_{window:012d}.csv.gz', dtype=dtypes, usecols=base, compression='gzip')
      test_df = pd.read_csv(f'files/test_df{_mid}_{window:012d}.csv.gz', dtype=dtypes, usecols=['click_id'] + list(filter(lambda x:x != 'is_attributed', base)), compression='gzip')
    except Exception as ex:
      print(ex)
      continue
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))
    
    target      = 'is_attributed'
    predictors  = list(filter(lambda x:x != 'is_attributed', base))
    #predictors  = ['app', 'device', 'os', 'channel', 'hour', 'day', 'qty', 'ip_app_count', 'ip_app_os_count', 'ip_os_hour_count', 'ip_os_app_hour_count', 'dh_f', 'hm', 'ci']
    categorical = ['app', 'device', 'os', 'channel', 'hour']

    print("Training...")
    params = {
      'boosting_type':    'gbdt',
      'objective':        'binary',
      'metric':           ['auc'],
      'learning_rate':    random.choice([0.100]),
      'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each
      'scale_pos_weight': 200, # because training data is extremely unbalanced 
      'num_leaves':       random.choice([7]),  # we should let it be smaller than 2^(max_depth)
      'max_depth':        random.choice([3]),  # -1 means no limit
      'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
      'max_bin':          100,  # Number of bucketed bin for feature values
      'subsample':        0.7,  # Subsample ratio of the training instance.
      'subsample_freq':   1,  # frequence of subsample, <=0 means no enable
      'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
      'subsample_for_bin': 200000,  # Number of samples for constructing bin
      'min_split_gain':   0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
      'reg_alpha':        0,  # L1 regularization term on weights
      'reg_lambda':       0,  # L2 regularization term on weights
      'verbose':          0,
    }
    obj = json.dumps( params, indent=2 )
    hash = hashlib.sha256(bytes(obj, 'utf8')).hexdigest()
    
    # ADHOC:train_df -> train_df + eval_df
    #train_df.append( val_df )

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )

    evals_results = {}
    bst1 = lgb.train(params, 
                     xgtrain, 
                     valid_sets            = [xgtrain, xgvalid], 
                     valid_names           = ['train','valid'], 
                     evals_result          = evals_results, 
                     num_boost_round       = 1000,
                     early_stopping_rounds = 50,
                     verbose_eval          = 10, 
                     feval                 = None)

    n_estimators = bst1.best_iteration

    feature_importances = bst1.feature_importance().tolist()
    feature_names = bst1.feature_name()

    importances = sorted(dict(zip(feature_names, feature_importances)).items(), key=lambda x:x[1]*-1)
    for feat, importance in importances:
      print(feat, importance)

    print("Model Report")
    print("n_estimators : ", n_estimators)
    auc = evals_results['valid']['auc'][n_estimators-1]
    print("auc:", auc)
    
    print('clear memory of dataset.')
    del xgtrain; del xgvalid 
    gc.collect()
    print("Predicting...")
    sub = pd.DataFrame()
    sub['click_id']      = test_df['click_id'].astype('int')
    sub['is_attributed'] = bst1.predict(test_df[predictors])
    print("writing...")
    sub.to_csv(f'submission_auc={auc:0.012f}_windows={window:012d}_est={n_estimators}_{hash}.csv',index=False)
    open(f'files/params/{hash}', 'w').write( obj )
    open(f'files/importances/{hash}', 'w').write( json.dumps(importances, indent=2) )
  
    print("done...")
