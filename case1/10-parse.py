import os

import sys

from pathlib import Path

import json
  
import numpy as np

from scipy import sparse

import pickle

import gzip
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

    # appはcategory
    app     = 'app:' + obj['app']
    # deviceはcategory
    device  = 'device:' + obj['device']
    # osはcategory
    os      = 'os:' + obj['os']
    # channelはcategory
    channel = 'channel:' + obj['channel']
   
    for feat in [app, device, os, channel]:
      if feat_index.get(feat) is None:
        feat_index[feat] = len(feat_index)
 
  # 手動で追加
  for i in range(10):
    feat_index[ f'ip_cat:{i}' ] = len(feat_index)
  feat_index[ 'ip_freq_lin' ] = len(feat_index)
  print(feat_index)

  json.dump(feat_index, fp=open('files/feat_index.json', 'w'), indent=2)

if '--step2' in sys.argv:

  
  ip_freq = {}
  ip_freq1 = json.load(fp=open('./files/click_ip_freq.json'))
  ip_freq2 = json.load(fp=open('./files/nclick_ip_freq.json'))
  for ip, freq in ip_freq1.items():
    if ip_freq.get(ip) is None:
      ip_freq[ip] = 0
    ip_freq[ip] += freq
  for ip, freq in ip_freq2.items():
    if ip_freq.get(ip) is None:
      ip_freq[ip] = 0
    ip_freq[ip] += freq

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

    ip_cat  = ip_freq[ obj['ip'] ] 
    ip_cat  = 'ip_cat' + len(str(ip_freq))

    ip_freq_lin = math.log( ip_freq[ obj['ip'] ] )
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

  data = gzip.compress( pickle.dumps( (Xs, ys) ) )
  open('files/data.pkl.gz', 'wb').write( data )
