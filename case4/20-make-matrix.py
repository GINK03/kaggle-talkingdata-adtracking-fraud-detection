import pickle

import os

import sys

import json

from scipy import sparse

import numpy as np

import gzip

from pathlib import Path

import random
# sparse matrixに変換してみる
if '--step1' in sys.argv:
  feat_index = json.load(fp=open('files/feat_index.json') )
  
  # write test
  fp_test = open('files/test_test.svm', 'w')
  for key, path in enumerate(sorted(Path('./files/').glob('test_*.pkl.gz'))):
    Xs, _ = pickle.loads( gzip.decompress( path.open('rb').read() ) )
    for index, xs in enumerate(Xs):
      if index%100000 == 0 and index != 0:
        print(f'make sparse now iter {index}@{key}/{path}')
        print(text)
        print(xs)
      ip_freq_lin, time_lin, app_freq_lin, ipXos_freq_lin, ipXapp_freq_lin, ipXappXos_freq_lin = xs 
      bs = { feat_index['ip_freq_lin']:ip_freq_lin }
      bs[ feat_index['time_lin'] ] = time_lin
      bs[ feat_index['app_freq_lin'] ] = app_freq_lin
      bs[ feat_index['ipXos_freq_lin'] ] = ipXos_freq_lin
      bs[ feat_index['ipXapp_freq_lin'] ] = ipXapp_freq_lin
      bs[ feat_index['ipXappXos_freq_lin'] ] = ipXappXos_freq_lin
      #for x in xs:
      #  bs[x] = 1.0

      bs = ' '.join( [f'{index}:{weight}' for index, weight in bs.items()] )
      text = f'0.0 {bs}\n'
      fp_test.write(text)
  
  # write train 
  fp_train = open('files/test_train.svm', 'w')
  fp_valid = open('files/test_valid.svm', 'w')

  dist_fp = {}
  fp_random = open('files/test_random.svm', 'w')
 
  for key, path in enumerate(sorted(Path('./files/').glob('train_valid_*.pkl.gz'))):
    Xs, ys = pickle.loads( gzip.decompress( path.open('rb').read() ) )
    # write train, valid
    for index, (xs, y) in enumerate(zip(Xs,ys)):
      if index%100000 == 0 and index != 0:
        print(f'make sparse now iter {index}@{key}/{path}')
        print(text)
      
      ip_freq_lin, time_lin, app_freq_lin, ipXos_freq_lin, ipXapp_freq_lin, ipXappXos_freq_lin = xs 
      bs = { feat_index['ip_freq_lin']:ip_freq_lin }
      bs[ feat_index['time_lin'] ] = time_lin
      bs[ feat_index['app_freq_lin'] ] = app_freq_lin
      bs[ feat_index['ipXos_freq_lin'] ] = ipXos_freq_lin
      bs[ feat_index['ipXapp_freq_lin'] ] = ipXapp_freq_lin
      bs[ feat_index['ipXappXos_freq_lin'] ] = ipXappXos_freq_lin
      #for x in xs:
      #  bs[x] = 1.0

      bs = ' '.join( [f'{index}:{weight}' for index, weight in bs.items()] )
      text = f'{y} {bs}\n'
      

      # key%Nでサンプルレートを決定する
      if key%20 == 0:
        fp_valid.write(text)
      else:
        fp_train.write(text)

        if random.random() < 0.2:
          fp_random.write( text )  
        dist = index%10
        if dist_fp.get(dist) is None:
          dist_fp[dist] = open(f'files/test_minitrain_{dist:09d}.svm', 'w')
        dist_fp[dist].write(text)
