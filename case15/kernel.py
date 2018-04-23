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

def DO(frm,to,fileno):
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

    # 差分
    train_df['wday'] = train_df['click_time'].dt.dayofweek.astype('uint8') 
    most_freq_hours_in_test_data = [4,5,9,10,13,14]
    least_freq_hours_in_test_data = [6, 11, 15]
    train_df['in_test_hh'] = ( 3
	    		 - 2 * train_df['hour'].isin( most_freq_hours_in_test_data )
			     - 1 * train_df['hour'].isin( least_freq_hours_in_test_data )).astype('uint8')
    gc.collect()
    
    def add_counts(df, cols):
      arr_slice = df[cols].values
      key = "_".join(cols)+"_count"
      if os.path.exists(f'{key}.pkl'):
        print(f'load {key}.pkl...')
        counts = pd.read_pickle(f'{key}.pkl') 
        df[key] = counts
        print(f'finish load {key}.pkl... ')
      else:
        print(f'create {key}.pkl...')
        unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(axis=0)+1),
                                       return_inverse=True, return_counts=True)
        df[key] = counts[unqtags]
        series = df[key]
        series.to_pickle(f'{key}.pkl')
        print(f'finish {key} save...')
    
    # 差分
    add_counts(train_df, ['ip'])
    add_counts(train_df, ['os', 'device'])
    add_counts(train_df, ['os', 'app', 'channel'])

    add_counts(train_df, ['ip', 'device'])
    add_counts(train_df, ['app', 'channel'])

    add_counts(train_df, ['ip', 'wday', 'in_test_hh'])
    add_counts(train_df, ['ip', 'wday', 'hour'])
    add_counts(train_df, ['ip', 'os', 'wday', 'hour'])
    add_counts(train_df, ['ip', 'app', 'wday', 'hour'])
    add_counts(train_df, ['ip', 'device', 'wday', 'hour'])
    add_counts(train_df, ['ip', 'app', 'os'])
    add_counts(train_df, ['wday', 'hour', 'app'])
    for i in range(0,naddfeat):
        if i==0: selcols=['ip', 'channel']; QQ=4;
        if i==1: selcols=['ip', 'channel']; QQ=5;
        if i==2: selcols=['ip', 'device', 'os', 'app']; QQ=4;
        if i==3: selcols=['ip', 'device', 'os', 'app']; QQ=5;
        if i==4: selcols=['ip', 'app', 'hour']; QQ=4; # ip,day,hour,4 微妙 -> 弱い
        if i==5: selcols=['ip', 'day', 'hour']; QQ=5;
        if i==6: selcols=['ip', 'app']; QQ=4;
        if i==7: selcols=['ip', 'hour', 'app']; QQ=5; # 5 ng; 検証中 -> 3
        if i==8: selcols=['ip', 'app', 'os']; QQ=4;
        if i==9: selcols=['ip', 'app', 'os']; QQ=5; # 弱い
        if i==10: selcols=['ip', 'device']; QQ=4; # 中くらい
        if i==11: selcols=['ip', 'device']; QQ=2; # 強い
        if i==12: selcols=['app', 'channel']; QQ=4; # 強い
        if i==13: selcols=['app', 'channel']; QQ=5; # 強い
        if i==14: selcols=['ip', 'os']; QQ=4; # 強い
        if i==15: selcols=['ip', 'os']; QQ=2; # 5ng
        if i==16: selcols=['ip', 'device', 'os', 'app']; QQ=2; # 4ng
        if i==17: selcols=['ip', 'device', 'os', 'app']; QQ=5; # 弱い -> 弱い
        if i==18: selcols=['ip', 'hour', 'os']; QQ=4; # 検証-> 18
        if i==19: selcols=['ip', 'hour', 'channel']; QQ=4; # 検証 -> 11
        if i==20: selcols=['ip', 'hour', 'app']; QQ=4; # 検証 -> 9
        if i==21: selcols=['ip', 'app', 'hour', 'os']; QQ=4; # 検証 -> 9
        print('selcols',selcols,'QQ',QQ)
        
        filename='X%d_%d_%d.csv'%(i,frm,to)
        
        if os.path.exists(filename):
            print(f'load {filename}...')
            if QQ==5: 
                gp=pd.read_csv(filename,header=None)
                train_df['X'+str(i)]=gp
            else: 
                gp=pd.read_csv(filename)
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            print(f'finish to load {filename}...')
        else:
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
            
            if not debug:
                 gp.to_csv(filename,index=False)
            
        del gp
        gc.collect()    

    print('doing nextClick')
    predictors=[]
   
    # start
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
        if not debug:
            print('saving')
            pd.DataFrame(QQ).to_csv(filename,index=False)
    train_df[new_feature] = QQ
    predictors.append(new_feature)
    train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature+'_shift')
    # end

    ## 繰り返し
    start = time.time()
    df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    df['nextClick2'] = (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time).astype(np.float32)
    df['prevClick2'] = (df.click_time - df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(+1)).astype(np.float32)
    print('Elapsed: {} seconds'.format(time.time() - start))
    ## end 繰り返し

    train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature+'_shift')
    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    
    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
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
      val2_df  = train_df[-250_0000:]
      val_df   = pd.read_pickle('files/val_df.pkl').fillna(-1.0)
      val_df   = val_df.append( val2_df )
      train_df = pd.read_pickle('files/train_df.pkl').fillna(-1.0)

    if '--merge' in sys.argv:
      train_df = train_df.append( val_df )
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))
    target = 'is_attributed'

    ignores = ['click_id', 'click_time', 'ip', 'is_attributed', 'category']
    #ignores.extend( ['x1', 'x7', 'x4', 'day', 'nextClick_shift', 'factrize'] )
    ignores.extend( ['ip_chl_ind'] )
    predictors = [ p for p in train_columns if p not in ignores ]
    #predictors.extend( ['ip_chl_ind'] )
    # regression test
    # predictors.extend( ['app_chl_conf', 'os_chl_conf' ] )
    categorical = list(filter( lambda x: x not in ignores,  ['app', 'device', 'wday', 'os', 'channel', 'hour', 'ip_chl_ind'] ) )
    print('predictors',predictors)

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    gc.collect()
    print("Training...")
    start_time = time.time()
    params = {
      'learning_rate'   : 0.20,
      #'is_unbalance': 'true', # replaced with scale_pos_weight argument
      'num_leaves'      : 7,  # 2^max_depth - 1
      'max_depth'       : 4,  # -1 means no limit
      'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
      'max_bin'         : 100,  # Number of bucketed bin for feature values
      'subsample'       : 0.7,  # Subsample ratio of the training instance.
      'subsample_freq'  : 1,  # frequence of subsample, <=0 means no enable
      'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
      'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
      'scale_pos_weight': 99.7 # because training data is extremely unbalanced 
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
    sub.to_csv(f'sub_it_{auc:012f}_{best_iteration:09d}.csv', index=False)

    print("done...")
    return sub

nrows=184903891-1
nchunk=40000000 + 1000_0000*6
val_size=500_0000

frm=nrows-75000000 - 1000_0000*6

to=frm+nchunk


naddfeat=22

if '--do' in sys.argv:
  sub=DO(frm,to,0)
if '--fun' in sys.argv:
  Fun()
