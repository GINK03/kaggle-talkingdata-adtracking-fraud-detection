
base = '''
task = train
data_random_seed={seed}

boosting_type = gbdt
objective = binary
metric = binary_logloss,auc
metric_freq = 1
is_training_metric = true
data = ./files/test_minitrain_{dist:09d}.svm
valid_data = ./files/test_valid.svm

tree_learner = serial
num_threads  = 16
feature_fraction = 0.8
bagging_freq = 5
bagging_fraction = 0.8
min_data_in_leaf = 50
min_sum_hessian_in_leaf = 5.0
is_enable_sparse = true
use_two_round_loading = false
is_save_binary_file = false
output_model = files/models/LightGBM_model_{dist:09d}.txt
num_machines = 1

learning_rate = 0.1
num_leaves = 7
max_depth = 4
min_child_weight= 0
colsample_bytree= 0.7
min_split_gain= 0
max_bin = 100
min_child_samples= 100
subsample= 0.7
subsample_freq= 1

is_unbalance=true
#scale_pos_weight= 99.7

#eval_freq=25
early_stopping_round=500
num_tree = 1500
'''

for dist in range(10):
  conf = base.format(seed=dist*7, dist=dist)
  print( conf )

  open(f'files/train_{dist:09d}.conf', 'w').write( conf )

pred = '''
task = predict
data = ./files/test_test.svm
input_model= files/models/LightGBM_model_{dist:09d}.txt
output_result = files/results/results_{dist:09d}.txt
'''
for dist in range(10):
  conf = pred.format(seed=dist*7, dist=dist)
  print( conf )

  open(f'files/pred_{dist:09d}.conf', 'w').write( conf )

