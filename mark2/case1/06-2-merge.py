
from pathlib import Path

import sys
import pickle, gzip
import gc
import os
import glob
import re

import concurrent.futures
paths = list(filter(lambda x: 'finish' not in x and re.search(r'c$', x) is None, glob.glob('var/chunks/*')))
size = len(paths)

pkls = sorted(glob.glob('var/*_count_all.pkl.gz'))
pkls_size = len(pkls)

keys_keyval = []
for name_index, name in enumerate(pkls):
  print(name_index, '@', pkls_size, name)
  key = str(name).split('/').pop().replace('_count_all.pkl.gz', '')
  print(key)
  key_val = pickle.loads( gzip.decompress(open(name, 'rb').read()) )
  keys = sorted(key.split('_'))
  keys_keyval.append( (keys, key_val) ) 

def pmap(init_name):
  if os.path.exists(f'{init_name}_finished'):
    return
  print(f'now converting {init_name}')
  init_name  = init_name
  input_name = init_name
  paths = sorted(Path('var').glob('*_count_all.pkl.gz'))
  size = len(paths)

  fp    = open(input_name)
  ft    = open(f'{input_name}_finished', 'w')
  if 'test_test_' in str(init_name):
    heads = open('var/head_test').read().split(',')
    print(heads)
  else:
    heads = open('var/head').read().split(',')

  for line in fp:
    vals    = line.strip().split(',') 
    text    = line.strip()
    obj     = dict(zip(heads,vals))
    for keys, key_val in keys_keyval:
      minikey = [v for k, v in sorted(obj.items(), key=lambda x:x[0]) if k in keys ]  
      minikey = '_'.join( minikey )
      val = key_val[ minikey ]  if key_val.get(minikey) else '0'
      #print(key, keys, minikey, val)
      text += f',{val}'
    ft.write( text + '\n' )

import random 
init_names = list(filter(lambda x: 'finish' not in x, glob.glob('var/chunks/*') ) )
random.shuffle(init_names)

[ pmap(x) for x in init_names ] 
