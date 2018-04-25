
import concurrent.futures


from pathlib import Path

import csv

import itertools

import pickle, gzip

import sys

import os

import random

ps =  sum( [list(itertools.combinations( ['ip','app','device','os','channel', 'wday', 'hour'], i ) ) for i in range(2,4)], [])
print(ps)

def pmap(arg):
  index, path = arg 
 
  try:
    print(index, path)
    if Path(f'var/02/{index:09d}_proceed').exists():
      print(f'already processed {index:09d}')
      return
    heads = open('var/head').read().split(',')
    it = csv.reader(path.open()) 

    key_val_freq = {}

    for i, vals in enumerate(it):
      if i%10000 == 0:
        print(f'now {i} @ {index} {path}')
        if Path(f'var/02/{index:09d}_proceed').exists():
          print(f'injection happen, processed {index:09d}')
          return
      obj = dict(zip(heads, vals)) 
      #print(obj)
      for p in ps:
        p = sorted(list(p))
        key = '_'.join(p)
        val = '_'.join([ v for k, v in sorted(obj.items(), key=lambda x:x[0]) if k in p ])
        if key_val_freq.get(key) is None:
          key_val_freq[key] = {}
        if key_val_freq[key].get(val) is None:
          key_val_freq[key][val] = 0
        key_val_freq[key][val] += 1

    for key, val_freq in key_val_freq.items():
      try:
        os.mkdir(f'var/02/{key}')
      except:
        ...
      data = gzip.compress(pickle.dumps(val_freq))
      open(f'var/02/{key}/{key}_{index:09d}.pkl.gz', 'wb').write( data )

    # flagを書き込んで終了
    Path(f'var/02/{index:09d}_proceed').open('w').write( 'finish' )
  except Exception as ex:
    print(ex)
if '1' in sys.argv:
  args = [(index, path) for index, path in enumerate(sorted(Path('var/chunks/').glob('*')))]
  random.shuffle(args)
  #pmap(args[0])
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map(pmap, args)

if '2' in sys.argv:
  key_val_freq = {}
  for index, path in enumerate(Path('var/02/').glob('*')):
    try:
      _key_val_freq = pickle.loads( gzip.decompress(path.open('rb').read()) )
    except Exception as ex:
      print(index, ex)
      continue
    print(index, path)
    for key, val_freq in _key_val_freq.items():
      if key_val_freq.get(key) is None:
        key_val_freq[key] = {}
      for val, freq in val_freq.items():
        if key_val_freq[key].get(val) is None:
          key_val_freq[key][val] = 0
        key_val_freq[key][val] += freq
  data = gzip.compress(pickle.dumps(key_val_freq))
  open(f'var/02_all.pkl.gz', 'wb').write( data )
