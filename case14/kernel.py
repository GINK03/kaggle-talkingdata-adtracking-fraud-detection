import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
#import matplotlib.pyplot as plt
import os
import sys
import json
debug=False
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

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
        'metric':metrics
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

def map_csv(arg):
  global train_df
  i, QQ, selcols, filename = arg
  print( i, QQ, selcols, filename )
  ...
  try:
    if QQ==0:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==1:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==2:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==3:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==4:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==5:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
        train_df['X'+str(i)]=gp.values
    train_df['X'+str(i)].to_csv(filename,index=False)
    del gp
    gc.collect()    
  except Exception as ex:
    print(ex)

train_df = None
def DO(frm,to,fileno):
    global train_df
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to, 'to-frm', to-frm)
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    print( train_df.columns.values)
    print('loading test data...')

    test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    print(f'Size of train = {len_train}')
    train_df=train_df.append(test_df)

    del test_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    
    gc.collect()
    
    # default singles 
    for dealtype, name, filters, bys, group in [ ('count', 'ip_tcount', ['ip','day','hour','channel'], ['ip','day','hour'], ['channel']), \
                                       ('count', 'ip_app_count', ['ip','app','channel'], ['ip','app'], ['channel']), \
                                       ('count', 'ip_app_os_count', ['ip','app', 'os','channel'], ['ip','app', 'os'], ['channel']), \
                                       ('count', 'ip_app_os_count', ['ip','app', 'os','channel'], ['ip','app', 'os'], ['channel']), \
                                       ('var', 'ip_day_chl_var_hour', ['ip','day', 'hour','channel'], ['ip','day', 'channel'], ['hour']), \
                                       ('var', 'ip_app_chl_var_day',     ['ip','day', 'app','channel'], ['ip','app', 'channel'], ['day']), \
                                       ('mean', 'ip_app_chl_mean_hour',  ['ip','day', 'hour','channel'], ['ip','app', 'channel'], ['hour'])  ]:
      if os.path.exists(f'{name}.csv'):
        ...
      else:
        if dealtype == 'count':
          print(f'grouping by {bys} {group} combination...')
          gp = train_df[filters].groupby(by=bys)[group].count().reset_index().rename(index=str, columns={group[0]: name})
          train_df = train_df.merge(gp, on=bys, how='left')
          train_df[ name ].to_csv(f'{name}.csv')
          del gp; gc.collect()
        if dealtype == 'var':
          print(f'grouping by : {name} of variance')
          gp = train_df[filters].groupby(by=bys)[group].var().reset_index().rename(index=str, columns={group[0]: name})
          train_df = train_df.merge(gp, on=bys, how='left')
          del gp;gc.collect()
        if dealtype == 'mean':
          print(f'grouping by : {name} of mean')
          gp = train_df[filters].groupby(by=bys)[group].mean().reset_index().rename(index=str, columns={group[0]: name})
          train_df = train_df.merge(gp, on=bys, how='left')
          del gp;gc.collect()
      
      gp = pd.read_csv(f'{name}.csv', header=None )
      train_df[name ]= gp
   
    args, tojoins = [], []
    for i in range(0,naddfeat):
        if i==0: selcols=['ip', 'channel']; QQ=2;
        if i==1: selcols=['ip', 'channel']; QQ=3;
        if i==2: selcols=['ip', 'channel']; QQ=4;
        if i==3: selcols=['ip', 'channel']; QQ=5;

        if i==4: selcols=['ip', 'device', 'os', 'app']; QQ=2;
        if i==5: selcols=['ip', 'device', 'os', 'app']; QQ=3;
        if i==6: selcols=['ip', 'device', 'os', 'app']; QQ=4;
        if i==7: selcols=['ip', 'device', 'os', 'app']; QQ=5;

        if i==8: selcols=['ip', 'day', 'hour']; QQ=2;
        if i==9: selcols=['ip', 'day', 'hour']; QQ=3;
        if i==10: selcols=['ip', 'day', 'hour']; QQ=4;
        if i==11: selcols=['ip', 'day', 'hour']; QQ=5;

        if i==12: selcols=['ip', 'app']; QQ=2;
        if i==13: selcols=['ip', 'app']; QQ=3;
        if i==14: selcols=['ip', 'app']; QQ=4;
        if i==15: selcols=['ip', 'app']; QQ=5;

        if i==16: selcols=['ip', 'app', 'os']; QQ=2;
        if i==17: selcols=['ip', 'app', 'os']; QQ=4;
        if i==18: selcols=['ip', 'app', 'os']; QQ=3;
        if i==19: selcols=['ip', 'app', 'os']; QQ=5;

        if i==20: selcols=['ip', 'device']; QQ=4;
        if i==21: selcols=['ip', 'device']; QQ=2;
        if i==22: selcols=['ip', 'device']; QQ=3; # 5ng
        if i==23: selcols=['ip', 'device']; QQ=5; # 5ng

        if i==24: selcols=['app', 'channel']; QQ=2;
        if i==25: selcols=['app', 'channel']; QQ=3;
        if i==26: selcols=['app', 'channel']; QQ=4;
        if i==27: selcols=['app', 'channel']; QQ=5;

        if i==28: selcols=['ip', 'os']; QQ=4;
        if i==29: selcols=['ip', 'os']; QQ=5;
        if i==30: selcols=['ip', 'os']; QQ=3; # 5ng
        if i==31: selcols=['ip', 'os']; QQ=2; # 5ng

        if i==32: selcols=['ip', 'device', 'os', 'app']; QQ=2; # 4ng
        if i==33: selcols=['ip', 'device', 'os', 'app']; QQ=5;
        if i==34: selcols=['ip', 'device', 'os', 'app']; QQ=3; # 4ng
        if i==35: selcols=['ip', 'device', 'os', 'app']; QQ=4;
        
        if i==36: selcols=['ip', 'os', 'app', 'channel']; QQ=4;
        if i==37: selcols=['ip', 'os', 'app', 'channel']; QQ=5;
        if i==38: selcols=['ip', 'os', 'app', 'channel']; QQ=3;
        if i==39: selcols=['ip', 'os', 'app', 'channel']; QQ=2;
        if i==40: selcols=['ip', 'os', 'app', 'channel']; QQ=1;

        if i==41: selcols=['ip', 'device', 'os', 'app', 'channel']; QQ=4;
        if i==42: selcols=['ip', 'device', 'os', 'app', 'channel']; QQ=5;
        if i==43: selcols=['ip', 'device', 'os', 'app', 'channel']; QQ=3;
        if i==43: selcols=['ip', 'device', 'os', 'app', 'channel']; QQ=2;

        if i==44: selcols=['device', 'os', 'app', 'ip']; QQ=2; 
        if i==45: selcols=['device', 'os', 'app', 'ip']; QQ=5;
        if i==46: selcols=['device', 'os', 'app', 'ip']; QQ=3; 
        if i==47: selcols=['device', 'os', 'app', 'ip']; QQ=4;

        print('selcols',selcols,'QQ',QQ)
        
        filename='X%d_%d_%d.csv'%(i,frm,to)
        
        if os.path.exists(filename):
          tojoins.append( (filename, QQ, i) )
          '''
            if QQ==5: 
                gp=pd.read_csv(filename,header=None)
                train_df['X'+str(i)]=gp
            else: 
                gp=pd.read_csv(filename)
                train_df['X'+str(i)]=gp
          '''
        else:
            arg =  (i, QQ, selcols, filename) 
            args.append( arg )
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as exe:
      exe.map( map_csv, args )
    if len(args) != 0:
      print('just only make csv')
      sys.exit(0)

    print('join extract futures')
    for tojoin in tojoins:
      filename, QQ, i = tojoin
      print( f'join {filename} {QQ} {i}' )
      gp=pd.read_csv(filename,header=None)
      train_df['X'+str(i)]=gp

    print('doing nextClick')
    predictors=[]
    
    new_feature = 'nextClick'
    filename='nextClick_%d_%d.csv'%(frm,to)

    if os.path.exists(filename):
        print('loading from save file')
        QQ=pd.read_csv(filename).values
    else:
        D=2**26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
            + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
        del(click_buffer)
        QQ= list(reversed(next_clicks))

        print('saving')
        pd.DataFrame(QQ).to_csv(filename,index=False)

    train_df[new_feature] = QQ
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature+'_shift')
    
    del QQ
    gc.collect()
    

    train_df.info()
    test_df  = train_df[len_train:]
    val_df   = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]
  
    #np.save( 'files/test_df', test_df.values ) 
    #np.save( 'files/val_df', val_df.values)
    #np.save( 'files/train_df', train_df.values)

    test_df.to_pickle('files/test_df.pkl')
    val_df.to_pickle('files/val_df.pkl')
    train_df.to_pickle('files/train_df.pkl')

    headers = [train_df.columns.values.tolist(), test_df.columns.values.tolist()] 
    json.dump( headers, fp=open('files/headers.json', 'w'), indent=2 ) 

def Fun():
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }
    train_columns, test_columns = json.load( fp=open('files/headers.json') ) 
    if '--numpy' in sys.argv:
      test_df = pd.DataFrame(np.load('files/test_df.npy'), columns=test_columns).infer_objects()
      val_df = pd.DataFrame(np.load('files/val_df.npy'), columns=train_columns).infer_objects()
      train_df = pd.DataFrame(np.load('files/train_df.npy'), columns=train_columns).infer_objects()
    else:
      test_df  = pd.read_pickle('files/test_df.pkl').fillna(-1.0)
      val_df   = pd.read_pickle('files/val_df.pkl').fillna(-1.0)
      train_df = pd.read_pickle('files/train_df.pkl').fillna(-1.0)
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))
    target = 'is_attributed'

    ignores = ['click_id', 'click_time', 'ip', 'is_attributed', 'category']
    ignores.extend( ['x1', 'x7', 'x4', 'day', 'nextClick_shift', 'factrize'] )
    ignores.extend( ['ip_chl_ind'] )
    predictors = [ p for p in train_columns if p not in ignores ]
    predictors.extend( ['epochtime'] )
    #predictors.extend( ['ip_chl_ind'] )
    # regression test
    # predictors.extend( ['app_chl_conf', 'os_chl_conf' ] )
    categorical = list(filter( lambda x: x not in ignores,  ['app', 'device', 'os', 'channel', 'hour', 'ip_chl_ind'] ) )
    print('predictors',predictors)

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    gc.collect()
    print("Training...")
    start_time = time.time()
    params = {
      'learning_rate': 0.20,
      #'is_unbalance': 'true', # replaced with scale_pos_weight argument
      'num_leaves': 7,  # 2^max_depth - 1
      'max_depth': 3,  # -1 means no limit
      'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
      'max_bin': 100,  # Number of bucketed bin for feature values
      'subsample': 0.7,  # Subsample ratio of the training instance.
      'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
      'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
      'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
      'scale_pos_weight': 200 # because training data is extremely unbalanced 
    }
    (bst,best_iteration, auc) = lgb_modelfit_nocv(params, 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()
    
    print('Plot feature importances...')
    #ax = lgb.plot_importance(bst, max_num_features=100)
    #plt.show()

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    
    print("writing...")
    sub.to_csv(f'sub_it_{auc:012f}.csv', index=False)

    print("done...")
    return sub

nrows=184903891-1
nchunk=40000000 + 1000_0000*4
val_size=2500000

frm=nrows-75000000 - 1000_0000*4

to=frm+nchunk

naddfeat=48

if '--do' in sys.argv:
  sub=DO(frm,to,0)
if '--fun' in sys.argv:
  Fun()
