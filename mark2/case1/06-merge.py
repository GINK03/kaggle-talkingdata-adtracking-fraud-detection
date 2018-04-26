
from pathlib import Path


import pickle, gzip
import gc

import glob

def pmap(init_name):
  init_name = init_name
  input_name = init_name
  try:
    paths = sorted(Path('var').glob('*_count_all.pkl.gz'))
    size = len(paths)
    for name_index, name in enumerate(paths):
      print(name_index, size, name)
      key = str(name).split('/').pop().replace('_count_all.pkl.gz', '')
      print(key)
      key_val = pickle.loads( gzip.decompress(name.open('rb').read()) )
      keys  = key.split('_')
      fp    = open(input_name)
      ft    = open(f'{input_name}c', 'w')
      if 'test_test_' in str(name):
        heads = open('var/head_test').read().split(',')
      else:
        heads = open('var/head').read().split(',')

      for line in fp:
        vals    = line.strip().split(',') 
        obj     = dict(zip(heads,vals))
        minikey = [v for k, v in sorted(obj.items(), key=lambda x:x[0]) if k in keys ]  
        minikey = '_'.join( minikey )
        val = key_val[ minikey ]  if key_val.get(minikey) else '0'
        #print(key, minikey, val)
        text = line.strip() + f',{val}'
        ft.write( text + '\n' )
      del key_val 
      gc.collect()
      # input_nameを更新
      input_name = f'{input_name}c'

    #最後にリネームして終了
    import re
    target_name = re.sub(r'c{2,}$', '_finish', input_name)
    print(target_name)
    Path(input_name).rename( Path(target_name) )
    import os
    #最後にcが最後についているデータを削除
    os.system(f'rm {init_name}c*')
  except Exception as ex:
    print(ex)

import concurrent.futures

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as exe:
  exe.map(pmap, glob.glob('var/chunks/test_test_*'))
