import os

import sys

from pathlib import Path

import json
  
import numpy as np

from scipy import sparse

import pickle

import gzip

import math

import concurrent.futures

if '--step1' in sys.argv: # feat indexを作成
  # train
  def _map_train(arg):
    key,path = arg
    feats = set()
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now iter {index}@{key}')
      obj  = json.loads(line.strip())
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
        feats.add(feat)
    return feats
  # test
  def _map_test(arg):
    key, path = arg
    feats = set()
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now iter {index}@{key}')
      obj  = json.loads(line.strip())
      # appはcategory
      app     = 'app:' + obj['app']
      # deviceはcategory
      device  = 'device:' + obj['device']
      # osはcategory
      os      = 'os:' + obj['os']
      # channelはcategory
      channel = 'channel:' + obj['channel']
      for feat in [app, device, os, channel]:
        feats.add(feat)
    return feats
 
  feat_index = {}
  args = [(index,path) for index, path in enumerate(Path('./files/data/').glob('train_*'))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    for _feats in exe.map(_map_train, args):
      for feat in _feats:
        if feat_index.get(feat) is None:
          feat_index[feat] = len(feat_index)
  args = [(index,path) for index, path in enumerate(Path('./files/data/').glob('test_*'))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    for _feats in exe.map(_map_test, args):
      for feat in _feats:
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
  feat_index = json.load(fp=open('files/feat_index.json'))
  
  # test data
  def _map_test(arg):
    key, path = arg
    Xs, ys = [], []
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now test iter {index}@{key}')
      obj = json.loads(line.strip())

      ip_cat  = ip_freq[ obj['ip'] ] 
      ip_cat  = 'ip_cat:{}'.format( len(str(ip_cat)) )
      ip_freq_lin = math.log( ip_freq[ obj['ip'] ] )
      # appはcategory
      app     = 'app:' + obj['app']
      # deviceはcategory
      device  = 'device:' + obj['device']
      # osはcategory
      os      = 'os:' + obj['os']
      # channelはcategory
      channel = 'channel:' + obj['channel']
     
      xs = [feat_index[feat] for feat in [ip_cat, app, device, os, channel]]
      xs.insert(0, ip_freq_lin) 
      Xs.append( xs )

    data = gzip.compress( pickle.dumps( (Xs, ys) ) )
    open(f'files/test_{key:09d}.pkl.gz', 'wb').write( data )
  args = [(index,path) for index, path in enumerate(Path('./files/data/').glob('test_*'))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map( _map_test, args )

  # train data
  def _map_train(arg):
    key, path = arg
    Xs, ys = [], []
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now train_valid iter {index}@{key}')
      obj = json.loads( line.strip() )

      is_attributed = obj['is_attributed']

      y = 1.0 if is_attributed == '1' else 0.0

      del obj['is_attributed'] 

      ip_cat  = ip_freq[ obj['ip'] ] 
      ip_cat  = 'ip_cat:{}'.format( len(str(ip_cat)) )
      ip_freq_lin = math.log( ip_freq[ obj['ip'] ] )
      # appはcategory
      app     = 'app:' + obj['app']
      # deviceはcategory
      device  = 'device:' + obj['device']
      # osはcategory
      os      = 'os:' + obj['os']
      # channelはcategory
      channel = 'channel:' + obj['channel']
     
      xs = [feat_index[feat] for feat in [ip_cat, app, device, os, channel]]
      xs.insert(0, ip_freq_lin) 
      Xs.append( xs ); ys.append( y )

    data = gzip.compress( pickle.dumps( (Xs, ys) ) )
    open(f'files/train_valid_{key:09d}.pkl.gz', 'wb').write( data )
  

  args = [(index,path) for index, path in enumerate(Path('./files/data/').glob('train_*'))]
  #_map_train(args[0]) 
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map( _map_train, args )
  
