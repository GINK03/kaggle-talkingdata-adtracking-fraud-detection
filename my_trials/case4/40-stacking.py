
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

import sys
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import hashlib

ha = hashlib.sha256(bytes('40-stacking.py','utf8')).hexdigest()

if '--simple' in sys.argv:
  if '--init' in sys.argv:
    print('start scan files')
    ys = []
    for line in open('./files/test_minitrain.svm'):
      y = float(line.strip().split(' ').pop(0))
      ys.append( y )
    ys = np.array(ys) 

    Xs = []
    for line in open('./files/result_test_minitrain'):
      x = float(line.strip())
      Xs.append( [x, 1] )
      #print(x)
    Xs = np.array(Xs) 

    yst = []
    for line in open('./files/test_valid.svm'):
      y = float(line.strip().split(' ').pop(0))
      yst.append( y )
    yst = np.array(yst) 

    Xst = []
    for line in open('./files/result_test_valid'):
      x = float(line.strip())
      Xst.append( [x, 1] )
    Xst = np.array(Xst)
    np.save( f'files/{ha}', (Xs, ys, Xst, yst) )
  
  Xs, ys, Xst, yst = np.load(f'files/{ha}.npy')

  print('finish scan files')
  model = LogisticRegression()
  model.fit(Xs, ys)
  
  yp  = model.predict(Xst)
  auc = roc_auc_score(yst, yp)
  print(f'auc={auc}')

  for ytrue, ypred in zip(yst.tolist(), yp.tolist()):
    print(ytrue, ypred)
