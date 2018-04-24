
import concurrent.futures


from pathlib import Path

import csv

import itertools

import pickle, gzip

import sys

ps =  sum( [list(itertools.combinations( ['ip','app','device','os','channel', 'wday', 'hour'], i ) ) for i in range(2,6)], [])
print(ps)

def pmap(arg):
  index, path = arg 
  
  print(index, path)
  heads = open('var/head').read().split(',')
  
  it = csv.reader(path.open()) 

  
  key_val_freq = {}
  for vals in it:
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
  data = gzip.compress(pickle.dumps(key_val_freq))
  open(f'var/02/{index:09d}.pkl.gz', 'wb').write( data )

if '1' in sys.argv:
  args = [(index, path) for index, path in enumerate(sorted(Path('var/chunks/').glob('*')))]
  #pmap(args[0])
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map(pmap, args)

if '2' in sys.argv:
  key_val_freq = {}
  for path in Path('var/02/').glob('*'):
    _key_val_freq = pickle.loads( gzip.decompress(path.open('rb').read()) )
    print(path)
    for key, val_freq in _key_val_freq.items():
      if key_val_freq.get(key) is None:
        key_val_freq[key] = {}
      for val, freq in val_freq.items():
        if key_val_freq[key].get(val) is None:
          key_val_freq[key][val] = 0
        key_val_freq[key][val] += freq
