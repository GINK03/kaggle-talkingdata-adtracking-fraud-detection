import pandas as pd
import time
import numpy as np
from   sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

path = 'inputs/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,144903891), nrows=40000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df  = train_df.append(test_df)

print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day']  = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

# # of clicks for each ip-day-hour combination
print('group by...')
gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
print('merge...')
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')

# # of clicks for each ip-app combination
print('group by...')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')

# # of clicks for each ip-app-os combination
print('group by...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')

print("vars and data type: ")
train_df['qty']             = train_df['qty'].astype('uint16')
train_df['ip_app_count']    = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

# # of clicks for each ip-day-hour combination
print('group by...')
gp = train_df[['ip', 'day', 'hour', 'os', 'channel']].groupby(by=['ip', 'day', 'hour', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_hour_count'})
print('merge...')
train_df = train_df.merge(gp, on=['ip','day','hour', 'os'], how='left')

# # of clicks for each ip-day-hour combination
print('group by...')
gp = train_df[['ip', 'os', 'app', 'day', 'hour', 'channel']].groupby(by=['ip', 'os', 'app', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_os_app_hour_count'})
print('merge...')
train_df = train_df.merge(gp, on=['ip', 'os', 'app', 'day','hour'], how='left')

train_df.head(20)
test_df  = train_df[len_train:]
val_df   = train_df[(len_train-3000000):len_train]
train_df = train_df[:(len_train-3000000)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target      = 'is_attributed'
predictors  = ['app', 'device', 'os', 'channel', 'hour', 'day', 'qty', 'ip_app_count', 'ip_app_os_count', 'ip_os_hour_count', 'ip_os_app_hour_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

print("Training...")
params = {
  'boosting_type': 'gbdt',
  'objective': 'binary',
  'metric':'auc',
  'learning_rate': 0.1,
  #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
  'scale_pos_weight':99, # because training data is extremely unbalanced 
  'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
  'max_depth': 3,  # -1 means no limit
  'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
  'max_bin': 100,  # Number of bucketed bin for feature values
  'subsample': 0.7,  # Subsample ratio of the training instance.
  'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
  'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
  'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
  'subsample_for_bin': 200000,  # Number of samples for constructing bin
  'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
  'reg_alpha': 0,  # L1 regularization term on weights
  'reg_lambda': 0,  # L2 regularization term on weights
  'verbose': 0,
}

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
                 valid_sets=[xgtrain, xgvalid], 
                 valid_names=['train','valid'], 
                 evals_result=evals_results, 
                 num_boost_round=800,
                 early_stopping_rounds=50,
                 verbose_eval=10, 
                 feval=None)

n_estimators = bst1.best_iteration
print("Model Report")
print("n_estimators : ", n_estimators)
print("auc:", evals_results['valid']['auc'][n_estimators-1])

print("Predicting...")
sub['is_attributed'] = bst1.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced99.csv',index=False)
print("done...")
