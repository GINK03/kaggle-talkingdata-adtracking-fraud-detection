
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
  
  df = pd.read_csv('var/train_nexts.csv', skiprows=range(1, 10000_0000)) #skiprows=range(1,17000_0000) )
  df = df.drop(['click_time', 'attributed_time'], axis=1)  
  dfv = df[-250_0000:]
  df = df[:-250_0000]

  print(df.info())
  columns =  df.columns.tolist()
  ignores = ['click_id', 'click_time', 'ip', 'is_attributed', 'category']
  predictors = [ p for p in columns if p not in ignores ]
  print('columns', columns )
  print('predictors', predictors)
  categorical = list(filter( lambda x: x not in ignores,  ['channel', 'device', 'os', 'app',  'wday', 'hour', 'day'] ) )
  print('categorical', categorical)
  params = {
    'learning_rate'   : 0.20,
    # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves'      : 7,  # 2^max_depth - 1
    'max_depth'       : 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin'         : 100,  # Number of bucketed bin for feature values
    'subsample'       : 0.7,  # Subsample ratio of the training instance.
    'subsample_freq'  : 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 99.7 # because training data is extremely unbalanced 
  }
  (bst,best_iteration, auc) = lgb_modelfit_nocv(params, 
                            df, 
                            dfv, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)
  from datetime import datetime
  now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  bst.save_model(f'files/model_auc={auc:012f}_time={now}.txt')

if '2' in sys.argv:
  dft = pd.read_csv('var/test.csv')
  
  print(dft.info())
  columns =  dft.columns.tolist()
  ignores = ['click_id', 'click_time', 'ip', 'is_attributed', 'category']
  predictors = [ p for p in columns if p not in ignores ]
  print('columns', columns )
  print('predictors', predictors)
  categorical = list(filter( lambda x: x not in ignores,  ['channel', 'device', 'os', 'app',  'wday', 'hour', 'day'] ) )
  print('categorical', categorical)
  
  bst = lgb.Booster(model_file='files/model_auc=00000.978603_time=2018-04-26 19:06:56.txt')
  sub = pd.DataFrame()
  sub['click_id'] = dft['click_id'].astype('int')
  sub['is_attributed'] = bst.predict(dft[predictors],num_iteration=bst.best_iteration)
  
  sub.to_csv(f'sub_it.csv', index=False)
