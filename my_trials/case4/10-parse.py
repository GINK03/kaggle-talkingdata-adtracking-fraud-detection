import os

import sys

from pathlib import Path

import json
  
import numpy as np

from scipy import sparse

import pickle

import gzip

import math

from datetime import datetime

import concurrent.futures

if '--step1' in sys.argv: # feat indexを作成
  # train
  def _map_train(arg):
    key,path = arg
    feats = set()
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now iter {index}@{key}/{path}')
      obj  = json.loads(line.strip())

      is_attributed = obj['is_attributed']
      del obj['is_attributed'] 

      # click timeをパース
      click_time = obj['click_time']
      #dtime = datetime.strptime(click_time, '%Y-%m-%d %H:%M:%S')
      #print(dtime.day, dtime.hour, dtime.minute)

      # appはcategory
      app     = 'app:' + obj['app']
      # deviceはcategory
      device  = 'device:' + obj['device']
      # osはcategory
      os      = 'os:' + obj['os']
      # channelはcategory
      #channel = 'channel:' + obj['channel']
      for feat in [app, device, os]:#, channel]:
        feats.add(feat)
    return feats
  # test
  def _map_test(arg):
    key, path = arg
    feats = set()
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now iter {index}@{key}/{path}')
      obj  = json.loads(line.strip())
      # appはcategory
      app     = 'app:' + obj['app']
      # deviceはcategory
      device  = 'device:' + obj['device']
      # osはcategory
      os      = 'os:' + obj['os']
      # channelはcategory
      #channel = 'channel:' + obj['channel']
      for feat in [app, device, os]:#, channel]:
        feats.add(feat)
    return feats
 
  feat_index = {}

  
  args = [(index,path) for index, path in enumerate(sorted(Path('./files/data/').glob('train_*')))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    for _feats in exe.map(_map_train, args):
      for feat in _feats:
        if feat_index.get(feat) is None:
          feat_index[feat] = len(feat_index)
  
  args = [(index,path) for index, path in enumerate(sorted(Path('./files/data/').glob('test_*')))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    for _feats in exe.map(_map_test, args):
      for feat in _feats:
        if feat_index.get(feat) is None:
          feat_index[feat] = len(feat_index)

  # 手動で追加
  for i in range(10):
    feat_index[ f'ip_cat:{i}' ] = len(feat_index)
  feat_index[ 'ip_freq_lin' ] = len(feat_index)
  
  # 日付の情報を追加
  for i in range(30):
    feat_index[ f'time_cat:{i}' ] = len(feat_index)
  feat_index[ 'time_lin' ] = len(feat_index)

  # freqsを追加
  feat_index[ f'app_freq_lin' ] = len(feat_index)
  feat_index[ f'ipXapp_freq_lin' ] = len(feat_index)
  feat_index[ f'ipXos_freq_lin' ] = len(feat_index)
  feat_index[ f'ipXappXos_freq_lin' ] = len(feat_index)
  print(feat_index)

  json.dump(feat_index, fp=open('files/feat_index.json', 'w'), indent=2)

if '--step2' in sys.argv:
  ip_freq, app_freq, ipXapp_freq, ipXos_freq, ipXappXos_freq = json.load(fp=open('./files/freqs.json'))
  feat_index = json.load(fp=open('files/feat_index.json'))
  
  # test data
  def _map_test(arg):
    key, path = arg
    Xs, ys = [], []
    try:
      for index, line in enumerate(path.open()):
        if index%100000 == 0:
          print(f'now test iter {index}@{key}/{path}')
        obj = json.loads(line.strip())

        # click timeをパース
        click_time = obj['click_time']
        dtime = datetime.strptime(click_time, '%Y-%m-%d %H:%M:%S')
        #print(dtime.day, dtime.hour, dtime.minute)
        time_lin = dtime.hour + dtime.minute/60.0
        
        ip      = obj['ip']
        ip_cat  = ip_freq[ obj['ip'] ] 
        ip_cat  = 'ip_cat:{}'.format( len(str(ip_cat)) )
        ip_freq_lin = math.log( ip_freq[ obj['ip'] ] )
        # appはcategory
        app     = obj['app']
        app_cat = app
        app_freq_lin = app_freq[ obj['app'] ]
        # deviceはcategory
        device_cat = obj['device']
        # osはcategory
        os      = obj['os']
        os_cat  = obj['os']

        ipXos_freq_lin = ipXos_freq[ f'{ip}_x_{os}' ] 
        ipXapp_freq_lin = ipXapp_freq[ f'{ip}_x_{app}' ] 
        ipXappXos_freq_lin = ipXappXos_freq[ f'{ip}_x_{app}_x_{os}' ] 
        # channelはcategory
        channel_cat = obj['channel']
       
        xs = [float(x) for x in [app_cat, device_cat, os_cat, channel_cat]]
        xs += [ip_freq_lin, time_lin, app_freq_lin, ipXos_freq_lin, ipXapp_freq_lin, ipXappXos_freq_lin ] 
        Xs.append( xs )

      Xs, ys = np.array(Xs), np.array(ys)
      data = gzip.compress( pickle.dumps( (Xs, ys) ) )
      open(f'files/test_{key:09d}.pkl.gz', 'wb').write( data )
    except Exception as ex:
      print(ex)
  args = [(index,path) for index, path in enumerate(sorted(Path('./files/data/').glob('test_*')))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map( _map_test, args )
  # train data
  def _map_train(arg):
    key, path = arg
    Xs, ys = [], []
    for index, line in enumerate(path.open()):
      try:
        if index%100000 == 0:
          print(f'now train_valid iter {index}@{key}/{path}')
        obj = json.loads( line.strip() )

        is_attributed = obj['is_attributed']

        y = 1.0 if is_attributed == '1' else 0.0

        del obj['is_attributed'] 
        
        # click timeをパース
        click_time = obj['click_time']
        dtime = datetime.strptime(click_time, '%Y-%m-%d %H:%M:%S')
        #print(dtime.day, dtime.hour, dtime.minute)
        time_lin = dtime.hour + dtime.minute/60.0
        
        ip      = obj['ip']
        ip_cat  = ip_freq[ obj['ip'] ] 
        ip_cat  = 'ip_cat:{}'.format( len(str(ip_cat)) )
        ip_freq_lin = math.log( ip_freq[ obj['ip'] ] )
        # appはcategory
        app     = obj['app']
        app_cat = app

        app_freq_lin = app_freq[ obj['app'] ]
        # deviceはcategory
        device_cat  = obj['device']
        # osはcategory
        os      = obj['os']
        os_cat  = obj['os']
        

        ipXos_freq_lin = ipXos_freq[ f'{ip}_x_{os}' ] 
        ipXapp_freq_lin = ipXapp_freq[ f'{ip}_x_{app}' ] 
        ipXappXos_freq_lin = ipXappXos_freq[ f'{ip}_x_{app}_x_{os}' ] 
        # channelはcategory
        channel_cat = obj['channel']

        #xs = [feat_index[feat] for feat in [app, device, os, channel]]
        #xs.insert(0, ip_freq_lin) 
        #xs.insert(0, time_lin) 
        #for feat in [app_cat, device_cat, os_cat]:
        #  xs[ feat_index[feat]] = 1.0
        xs = [float(x) for x in [app_cat, device_cat, os_cat, channel_cat]]
        xs += [ip_freq_lin, time_lin, app_freq_lin, ipXos_freq_lin, ipXapp_freq_lin, ipXappXos_freq_lin ] 
        Xs.append( xs ); ys.append( y )
      except Exception as ex:
        print(ex)

    Xs, ys = np.array(Xs), np.array(ys)
    data = gzip.compress( pickle.dumps( (Xs, ys) ) )
    open(f'files/train_valid_{key:09d}.pkl.gz', 'wb').write( data )
  

  args = [(index,path) for index, path in enumerate(sorted(Path('./files/data/').glob('train_*')))]
  #_map_train(args[0]) 
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map( _map_train, args )
  
