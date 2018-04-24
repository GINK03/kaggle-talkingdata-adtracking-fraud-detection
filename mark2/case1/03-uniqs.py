
import concurrent.futures

from pathlib import Path

import csv

import itertools

import pickle, gzip

import sys

import random
ps =  sum( [list(itertools.permutations( ['ip','app','device','os','channel'], i ) ) for i in range(3,4)], [])
print(ps)
def pmap(arg):
  index, path = arg 
  
  print(index, path)
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
  data = gzip.compress(pickle.dumps(key_group_space))
  open(f'var/03/{index:09d}.pkl.gz', 'wb').write( data )

if '1' in sys.argv:
  args = [(index, path) for index, path in enumerate(sorted(Path('var/chunks/').glob('*')))]
  #pmap(args[0])
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map(pmap, args)
