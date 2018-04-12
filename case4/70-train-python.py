
import gzip
import pickle
import sys
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.datasets import load_svmlight_file

from pathlib import Path
Xs, ys = pickle.loads(gzip.decompress(open('./files/train_valid_000000001.pkl.gz', 'rb').read()))
Xsv, ysv = pickle.loads(gzip.decompress(open('./files/train_valid_000000004.pkl.gz', 'rb').read()))

print(Xs.shape)
params = {}
params['learning_rate'] = 0.02
params['n_estimators'] = 650
params['max_bin'] = 10
params['subsample'] = 0.8
params['subsample_freq'] = 10
params['colsample_bytree'] = 0.8   
params['min_child_samples'] = 500
params['seed'] = 99
params['silent'] = False
params['eval_metric'] = ['auc', 'binary_logloss']
model = LGBMClassifier(**params)
model.fit(Xs, ys, verbose=True, eval_set=(Xsv, ysv), eval_metric=['auc', 'binary_logloss'])


fp = open(f'files/submission.csv', 'w')
fp.write('click_id,is_attributed\n')
count = 0
for path in sorted(Path('files/').glob('test_*.pkl.gz')):
  print(path)
  Xst, yst = pickle.loads(gzip.decompress(open(f'{path}', 'rb').read()))
  yp = model.predict_proba(Xst)[:,1]
  for y in yp.tolist():
    fp.write(f'{count},{y:0.09f}\n')
    count += 1

