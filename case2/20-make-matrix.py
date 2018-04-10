import pickle

import os

import sys

import json

from scipy import sparse

import numpy as np

import gzip
# sparse matrixに変換してみる
if '--step1' in sys.argv:
  feat_index = json.load(fp=open('files/feat_index.json') )
  Xs, ys = pickle.loads( gzip.decompress(open('files/train_valid.pkl.gz', 'rb').read()) )

  size = len(Xs)
  split = int(len(Xs)*0.8)
  width = len(feat_index) 
  print(size)

  fp_train = open('files/test_train.svm', 'w')
  fp_valid = open('files/test_valid.svm', 'w')
  
  # write train, valid
  for index, (xs, y) in enumerate(zip(Xs,ys)):
    if index%100000 == 0 and index != 0:
      print(f'make sparse now iter {index}')
      print(text)
    #print(xs)
    ip_freq = xs.pop(0)
    
    bs = { feat_index['ip_freq_lin']:ip_freq }
    for x in xs:
      bs[x] = 1.0

    bs = ' '.join( [f'{index}:{weight}' for index, weight in bs.items()] )
    text = f'{y} {bs}\n'
    if index >= split:
      fp_valid.write(text)
    else:
      fp_train.write(text)
   
  # write test
  fp_test = open('files/test_test.svm', 'w')
  Xs, _ = pickle.loads( gzip.decompress(open('files/test.pkl.gz', 'rb').read()) )
  for index, xs in enumerate(Xs):
    if index%100000 == 0 and index != 0:
      print(f'make sparse now iter {index}')
      print(text)
    #print(xs)
    ip_freq = xs.pop(0)
    
    bs = { feat_index['ip_freq_lin']:ip_freq }
    for x in xs:
      bs[x] = 1.0

    bs = ' '.join( [f'{index}:{weight}' for index, weight in bs.items()] )
    text = f'0.0 {bs}\n'
    fp_test.write(text)
