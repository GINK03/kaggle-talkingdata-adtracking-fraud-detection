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
  data = pickle.loads( gzip.decompress(open('files/data.pkl', 'rb').read()) )

  size = len(data)
  width = len(feat_index) 
  print(size)

  mat = sparse.dok_matrix((size, width), dtype=np.int8)

  for index, one in enumerate(data):
    if index%1000 == 0:
      print(f'now iter {index}')
    mat[ index, one ] = 1.0

  mat = mat.transpose().tocsr()

  mat = gzip.compress(pickle.dump(mat))
  open('files/mat.pkl.gz', 'wb').write( mat )
