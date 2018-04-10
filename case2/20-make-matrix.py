import pickle

import os

import sys

import json

from scipy import sparse

import numpy as np

import gzip

from pathlib import Path
# sparse matrixに変換してみる
if '--step1' in sys.argv:
  feat_index = json.load(fp=open('files/feat_index.json') )
  
  fp_train = open('files/test_train.svm', 'w')
  fp_valid = open('files/test_valid.svm', 'w')
 
  for key, path in enumerate(Path('./files/').glob('train_valid_*')):
    Xs, ys = pickle.loads( gzip.decompress( path.open('rb').read() ) )
    # write train, valid
    for index, (xs, y) in enumerate(zip(Xs,ys)):
      if index%100000 == 0 and index != 0:
        print(f'make sparse now iter {index}@{key}')
        print(text)
      #print(xs)
      ip_freq = xs.pop(0)
      
      bs = { feat_index['ip_freq_lin']:ip_freq }
      for x in xs:
        bs[x] = 1.0

      bs = ' '.join( [f'{index}:{weight}' for index, weight in bs.items()] )
      text = f'{y} {bs}\n'
      if key%5 != 0:
        fp_valid.write(text)
      else:
        fp_train.write(text)
   
  # write test
  fp_test = open('files/test_test.svm', 'w')
  for key, path in enumerate(Path('./files/').glob('test_*')):
    Xs, _ = pickle.loads( gzip.decompress( path.open('rb').read() ) )
    for index, xs in enumerate(Xs):
      if index%100000 == 0 and index != 0:
        print(f'make sparse now iter {index}@{key}')
        print(text)
      #print(xs)
      ip_freq = xs.pop(0)
      
      bs = { feat_index['ip_freq_lin']:ip_freq }
      for x in xs:
        bs[x] = 1.0

      bs = ' '.join( [f'{index}:{weight}' for index, weight in bs.items()] )
      text = f'0.0 {bs}\n'
      fp_test.write(text)
