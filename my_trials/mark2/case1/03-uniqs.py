
import concurrent.futures

from pathlib import Path

import csv

import itertools

import pickle, gzip

import sys

import random

import os
ps =  sum( [list(itertools.permutations( ['ip','app','device','os','channel'], i ) ) for i in range(3,4)], [])
ps = [sorted(p) for p in ps]
print(ps)
def pmap(arg):
  index, path = arg 
  
  try:
    print(index, path)
    if os.path.exists(f'var/03/{index:09d}_proceed'):
      return
    heads = open('var/head').read().split(',')
     
    it = csv.reader(path.open()) 
    
    key_group_space = {}
    for vals in it:
      obj = dict(zip(heads, vals)) 
      #print(obj)
      for p in ps:
        p = list(p)
        group, by = p[:-1], p[-1]
        key = '_'.join(p) + '_uniq'
        group = '_'.join( [v for k, v in sorted(obj.items(), key=lambda x:x[0]) if k in group] )
        val =  obj[by]
        
        #if random.random() < 0.00001:
        #  print(key, group, val)
        if key_group_space.get(key) is None:
          key_group_space[key] = {}
        if key_group_space[key].get(group) is None:
          key_group_space[key][group] = set()
        key_group_space[key][group].add( val )

    for key, group_space in key_group_space.items():
      try:
        os.mkdir(f'var/03/{key}')
      except:
        ...
      data = gzip.compress(pickle.dumps(group_space))
      open(f'var/03/{key}/{index:09d}.pkl.gz', 'wb').write( data )
    Path(f'var/03/{index:09d}_proceed').open('w').write( 'finish' )
  except Exception as ex:
    print(ex)
if '1' in sys.argv:
  import glob
  args = [(index, Path(path)) for index, path in enumerate(sorted(filter(lambda x:'finished' not in x , glob.glob('var/chunks/*'))))]
  random.shuffle(args)
  #pmap(args[0])

  th = 16
  if 'th4' in sys.argv:
    th = 4
  if 'th2' in sys.argv:
    th = 2
  with concurrent.futures.ProcessPoolExecutor(max_workers=th) as exe:
    exe.map(pmap, args)
