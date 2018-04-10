import json
import hashlib
num_treess = [i for i in range(300,1000,50)]
num_leavess = [i for i in range(63, 100)]
fractions = [i/100.0 for i in range(10,100)]
bagging_freqs = [i for i in range(5,100)]
min_data_in_leafs = [i for i in range(50,100)]
min_sum_hessian_in_leafs = [i/10.0 for i in range(50,100)]
learning_rates = [i/1000.0 for i in range(1,15)]

for learning_rate in learning_rates:
  for num_tree in num_treess:
    for num_leaves in num_leavess:
      for fraction in fractions:
        for bagging_freq in bagging_freqs:
          for min_data_in_leaf in min_data_in_leafs:
            for min_sum_hessian_in_leaf in min_sum_hessian_in_leafs:
              data = [learning_rate, num_tree, num_leaves, fractions, bagging_freq, min_data_in_leaf, min_sum_hessian_in_leaf] 
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
              num_trees = {num_tree}
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
            #print(conf)
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

