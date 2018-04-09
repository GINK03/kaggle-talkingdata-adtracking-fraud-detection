import os

import sys

from pathlib import Path

import json
  
import numpy as np

from scipy import sparse

import pickle
if '--step1' in sys.argv: # feat indexを作成
  fp = open('../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/mnt/ssd/kaggle-talkingdata2/competition_files/train.csv')

  head = next(fp).strip().split(',')

  feat_index = {}
  for index, line in enumerate(fp):
    if index%100000 == 0:
      print(f'now iter {index}')
    val = line.strip().split(',')
    obj = dict( zip(head, val) )
    is_attributed = obj['is_attributed']
    del obj['is_attributed'] 

    # ipはcategory
    ip      = 'ip:' + obj['ip']
    # appはcategory
    app     = 'app:' + obj['app']
    # deviceはcategory
    device  = 'device:' + obj['device']
    # osはcategory
    os      = 'os:' + obj['os']
    # channelはcategory
    channel = 'channel:' + obj['channel']
    
    for feat in [ip, app, device, os, channel]:
      if feat_index.get(feat) is None:
        feat_index[feat] = len(feat_index)

  print(feat_index)

  json.dump(feat_index, fp=open('files/feat_index.json', 'w'), indent=2)

if '--step2' in sys.argv:


  fp = open('../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/mnt/ssd/kaggle-talkingdata2/competition_files/train.csv')

  head = next(fp).strip().split(',')

  feat_index = json.load(fp=open('files/feat_index.json'))
  
  Xs, ys = [], []
  for index, line in enumerate(fp):
    if index%100000 == 0:
      print(f'now iter {index}')
    val = line.strip().split(',')
    obj = dict( zip(head, val) )
    is_attributed = obj['is_attributed']

    y = 1.0 if is_attributed == '1' else 0.0

    del obj['is_attributed'] 

    # ipはcategory
    ip      = 'ip:' + obj['ip']
    # appはcategory
    app     = 'app:' + obj['app']
    # deviceはcategory
    device  = 'device:' + obj['device']
    # osはcategory
    os      = 'os:' + obj['os']
    # channelはcategory
    channel = 'channel:' + obj['channel']
   
    xs = [feat_index[feat] for feat in [ip, app, device, os, channel]]
     
    Xs.append( xs ); ys.append( y )

  pickle.dump( (Xs, ys), open('files/data.pkl', 'wb'))
