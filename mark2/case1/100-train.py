
import pandas as pd
import sys
import lightgbm as lgb
import json
def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'verbose': 0,
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])
    auc = evals_results['valid']['auc'][bst1.best_iteration-1]
    
    feature_importances = bst1.feature_importance().tolist()
    feature_names = bst1.feature_name()

    importances = sorted(dict(zip(feature_names, feature_importances)).items(), key=lambda x:x[1]*-1)
    for feat, importance in importances:
      print('importances', feat, importance)
    
    open(f'files/importances_auc={auc}', 'w').write( json.dumps(importances, indent=2) )

    return (bst1,bst1.best_iteration, auc)

if '1' in sys.argv:
  target = 'is_attributed'
 
  if 'simple' in sys.argv:
    print('load csv')
    #usecols = [ x[0] for x in sorted( json.load(open('files/base20180428')), key=lambda x:x[1] )[-20:] ] 
    usecols = 'ip,app,device,os,channel,click_time,attributed_time,is_attributed,hour,var/app_ip_os_count_all,var/app_ip_wday_count_all,var/app_wday_count_all,var/device_hour_ip_count_all,var/device_ip_count_all,var/device_ip_os_count_all,var/device_ip_wday_count_all,var/hour_ip_count_all,var/hour_ip_wday_count_all,var/ip_os_wday_count_all,var/ip_wday_count_all,ip_os_app_device_nextclick,ip_os_app_nextclick,ip_os_app_device_nextnextclick,ip_os_app_device_prevclick'.split(',')
    print( usecols )
    df = pd.read_csv('var/train_nexts.csv', usecols=usecols)# skiprows=range(1, 8000_0000)) #skiprows=range(1,17000_0000) )
    print('finish csv')
  else:
    import dask.dataframe as dd
    import dask.multiprocessing
    import glob
    import random
    print('load csv with dask')
    usecols = [ x[0] for x in json.loads(open('files/base2018430').read()) ][:31] 
    usecols.extend(['attributed_time', 'is_attributed'])
    dft = dd.read_csv(sorted(glob.glob('var/chunk_alot/shrink_train_nexts_*.csv'))[1000:-50],  # read in parallel
               sep=',', 
               parse_dates=['attributed_time'], 
               blocksize=1000000,
               usecols=usecols,
               )
    print(sorted(glob.glob('var/chunk_alot/shrink_train_nexts_*.csv'))[-50:])
    dfv = dd.read_csv(sorted(glob.glob('var/chunk_alot/shrink_train_nexts_*.csv'))[-50:],  # read in parallel
               sep=',', 
               parse_dates=['attributed_time'], 
               blocksize=1000000,
               usecols=usecols,
               )
    print('collect and convert pandas-dataframe...')
    dft = dft.drop(['attributed_time'], axis=1)
    #dft = dft.drop(['ip', 'click_time'], axis=1)  
    dft = dft.compute(get=dask.multiprocessing.get)
    dfv = dfv.drop(['attributed_time'], axis=1)
    #dfv = dfv.drop(['ip', 'click_time'], axis=1)  
    dfv = dfv.compute(get=dask.multiprocessing.get)
    print('finish collect and convert pandas-dataframe...')

  try:
    df = df.drop(['attributed_time'], axis=1)
    df = df.drop(['ip', 'click_time'], axis=1)  
  except Exception as ex:
    print(ex)
    ...
  
  try:
    columns =  df.columns.tolist()
  except NameError as ex:
    columns =  dft.columns.tolist()
  #dfv0 = df[:250_0000]

  predictors = [x for x in columns if x not in ['is_attributed', 'ip'] ]
  print('predictors', predictors)

  categorical = ['channel', 'app', 'os', 'device', 'hour'] 
  #for cat in categorical:
  #  df[cat] = df[cat].astype('category')
  #print(df.info())
  if 'random' in sys.argv:
    print('use random cropping')
    from sklearn.model_selection import train_test_split
    dft, dfv = train_test_split(df, test_size=0.1)
  else:
    print('use time separate')
    #split = len(df) - 500_0000
    #dfv = df[split:len(df)-1]
    #dft = df[0:split - 5000_0000]
    # noising...
    nparam = {'channel': 0.8, 'os':0.8, 'app':0.6}
    if 'noise' in sys.argv:
      def noise(prob):
        def _noise(x):
          if random.random() < prob:
            return x
          else:
            return 0.0
        return _noise
      print('noising...')

      dft[ 'channel' ] = dft[ 'channel' ].apply(noise(nparam['channel']))
      dft[ 'os' ] = dft[ 'os' ].apply(noise(nparam['os']))
      dft[ 'app' ] = dft[ 'app' ].apply(noise(nparam['app']))
      dfv[ 'channel' ] = dfv[ 'channel' ].apply(noise(nparam['channel']))
      dfv[ 'os' ] = dfv[ 'os' ].apply(noise(nparam['os']))
      dfv[ 'app' ] = dfv[ 'app' ].apply(noise(nparam['app']))
      print('finish noising...')
  #dfv = df[len(df) - 250_0000:]
  #dft = df[:len(df) - 250_0000]
  print('categorical', categorical)
  print('columns', columns )
  print('total-szie', len(dft)+len(dfv))
  print('train-size', len(dft))
  print('test-size', len(dfv))
  params = {
    'learning_rate'   : 0.20,
    # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves'      : 7,  # 2^max_depth - 1
    'max_depth'       : 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin'         : 100,  # Number of bucketed bin for feature values
    'subsample'       : 0.77,  # Subsample ratio of the training instance.
    'subsample_freq'  : 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.77,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 300 # because training data is extremely unbalanced 
  }
  (bst,best_iteration, auc) = lgb_modelfit_nocv(params, 
                            dft, 
                            dfv, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=140, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)
  from datetime import datetime
  now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  bst.save_model(f'files/model_auc={auc:012f}_time={now}_best={best_iteration}.txt')
  open(f'files/noise_auc={auc:012f}_time={now}_best={best_iteration}.json', 'w').write( json.dumps(nparam, indent=2) )

if '2' in sys.argv:
  usecols = 'click_id,ip,app,device,os,channel,click_time,hour,var/app_ip_os_count_all,var/app_ip_wday_count_all,var/app_wday_count_all,var/device_hour_ip_count_all,var/device_ip_count_all,var/device_ip_os_count_all,var/device_ip_wday_count_all,var/hour_ip_count_all,var/hour_ip_wday_count_all,var/ip_os_wday_count_all,var/ip_wday_count_all,ip_os_app_device_nextclick,ip_os_app_nextclick,ip_os_app_device_nextnextclick,ip_os_app_device_prevclick'.split(',')
  
  real = next(open('var/test_nexts.csv')).strip().split(',')

  for u in usecols:
    if u not in real:
      print(u)
  #sys.exit()
  dft = pd.read_csv('var/test_nexts.csv') #, usecols=usecols)
  try: dft = dft.drop(['attributed_time'], axis=1);
  except: ...;
  try: dft = dft.drop(['ip', 'click_time'], axis=1);
  except: ...;
  categorical = ['channel', 'os', 'device', 'app',  'hour'] 
  for cat in categorical:
    dft[cat] = dft[cat].astype('category')

  print(dft.info())
  columns =  dft.columns.tolist()
  ignores = ['click_id', 'click_time', 'ip', 'is_attributed']
  predictors = [ p for p in columns if p not in ignores ]
  print('columns', columns )
  print('predictors', predictors)
  print('categorical', categorical)
  
  model_file = 'files/model_auc=00000.989086_time=2018-04-30 04:47:14_best=247.txt'
  sufix = model_file.split('/').pop().replace(' ', '_')

  bst = lgb.Booster(model_file=model_file)
  sub = pd.DataFrame()
  sub['click_id'] = dft['click_id'].astype('int')
  sub['is_attributed'] = bst.predict(dft[predictors],num_iteration=bst.best_iteration)
  
  sub.to_csv(f'sub_it_{sufix}.csv', index=False)
