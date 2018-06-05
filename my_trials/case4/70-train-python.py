
import gzip
import pickle
import sys
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.datasets import load_svmlight_file
import random
from pathlib import Path
import hashlib
import numpy as np
import os
if '--train' in sys.argv:
  paths = [path for path in Path('./files').glob('train_valid_*.pkl.gz')]

  for trial in range(1000):
    paths = random.sample(paths, 4)
    Xs, ys = [], []
    for path_train in paths[:3]:
      _Xs, _ys = pickle.loads(gzip.decompress(open(f'{path_train}', 'rb').read()))
      Xs.append( _Xs ) 
      ys.append( _ys )
    
    Xs = np.concatenate( tuple(Xs), axis=0 )
    ys = np.concatenate( tuple(ys), axis=0 )

    path_valid = paths[-1]
    print(paths[:3], paths[-1])
    Xsv, ysv = pickle.loads(gzip.decompress(open(f'{path_valid}', 'rb').read()))

    print(Xs.shape)
    params = {}
    params['learning_rate']     = 0.02
    params['n_estimators']      = 650
    params['max_bin']           = 10
    params['subsample']         = 0.8
    params['subsample_freq']    = 10
    params['colsample_bytree']  = 0.8   
    params['min_child_samples'] = 500
    params['seed']              = 99
    params['silent']            = False
    params['eval_metric']       = ['auc', 'binary_logloss']
    model = LGBMClassifier(**params)
    model.fit(Xs, ys, categorical_feature=[0,1,2,3], verbose=True, eval_set=(Xsv, ysv), early_stopping_rounds=400, eval_metric=['auc', 'binary_logloss'])
    
    data = gzip.compress(pickle.dumps(model))
    ha = hashlib.sha256(data).hexdigest()
    open(f'files/models/{ha}', 'wb').write( data )

if '--predict' in sys.argv:
  for model_path in Path('./files/models').glob('*'):
    ha = f'{model_path}'.split('/').pop().strip()

    if Path(f'./files/predicts/{ha}').exists():
      print(f'already processed {ha}')
      continue
    model = pickle.loads(gzip.decompress(model_path.open('rb').read()))

    yps = []
    for path in sorted(Path('files/').glob('test_*.pkl.gz')):
      print(model_path, path)
      Xst, yst = pickle.loads(gzip.decompress(open(f'{path}', 'rb').read()))
      yp = model.predict_proba(Xst)[:,1]
      yps.append( yp )
    yps = np.concatenate( tuple(yps), axis=0 )
    print(yps.shape)
    
    data = gzip.compress( pickle.dumps( yps ) )
    open(f'./files/predicts/{ha}', 'wb').write( data )

if '--ensemble' in sys.argv:
  # bind train-data
  yps = []
  for path in sorted(Path('./files/predicts/').glob('*')):
    yp = pickle.loads(gzip.decompress(open(f'{path}', 'rb').read()))

    size = yp.shape[0]
    yps.append( yp.reshape(size, 1) )
  yps = np.concatenate( tuple(yps), axis=1 )
  print(yps.shape)
  yps = np.mean(yps, axis=1)
      
  fp = open(f'files/submission.csv', 'w')
  fp.write('click_id,is_attributed\n')
  for count, y in enumerate(yps.tolist()):
    fp.write(f'{count},{y:0.09f}\n')
    count += 1
