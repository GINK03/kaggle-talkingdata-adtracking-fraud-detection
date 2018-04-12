import json
import hashlib

num_leavess = [63,70,80, 100]
fractions = [0.10, 0.50, 0.60, 0.100]
bagging_freqs = [5,10]
min_data_in_leafs = [50,100]
min_sum_hessian_in_leafs = [5,6,7,8]
learning_rates = [0.001, 0.025, 0.035]

for learning_rate in learning_rates:
  for num_leaves in num_leavess:
    for fraction in fractions:
      for bagging_freq in bagging_freqs:
        for min_data_in_leaf in min_data_in_leafs:
          for min_sum_hessian_in_leaf in min_sum_hessian_in_leafs:
            data = [learning_rate, num_leaves, fractions, bagging_freq, min_data_in_leaf, min_sum_hessian_in_leaf] 
            data = json.dumps(data)
            ha = hashlib.sha256(bytes(data, 'utf8')).hexdigest()
            conf = f'''
            task = train
            boosting_type = gbdt
            objective = binary
            metric = binary_logloss,auc
            metric_freq = 1
            is_training_metric = true
            max_bin = 255
            data = ./files/test_train.svm
            valid_data = ./files/test_valid.svm
            num_trees = 750
            learning_rate = {learning_rate}
            num_leaves = {num_leaves}
            tree_learner = serial
            num_threads  = 16
            feature_fraction = {fraction}
            bagging_freq = {bagging_freq}
            bagging_fraction = 0.8
            min_data_in_leaf = {min_data_in_leaf}
            min_sum_hessian_in_leaf = {min_sum_hessian_in_leaf}
            is_enable_sparse = true
            use_two_round_loading = false
            is_save_binary_file = false
            output_model = files/models/{ha}.txt
            num_machines = 1
            '''
          open(f'files/confs/{ha}', 'w').write( conf )
          print(conf)
defconf = '''
task = train
boosting_type = gbdt
objective = binary
metric = binary_logloss,auc
metric_freq = 1
is_training_metric = true
max_bin = 255
data = ./files/test_train.svm
valid_data = ./files/test_valid.svm
num_trees = 300
learning_rate = 0.005
num_leaves = 63
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
output_model = LightGBM_model.txt
num_machines = 1
'''

