
from pathlib import Path


import pickle, gzip


input_name = 'var/chunks/000000000'
for name_index, name in enumerate(sorted(Path('var').glob('*_count_all.pkl.gz'))):
  print(name)
  key = str(name).split('/').pop().replace('_count_all.pkl.gz', '')
  print(key)
  key_val = pickle.loads( gzip.decompress(name.open('rb').read()) )
  keys = key.split('_')
 
  fp    = open(input_name)
  ft    = open(f'{input_name}c', 'w')
  heads = open('var/head').read().split(',')
  for line in fp:
    vals    = line.strip().split(',') 
    obj     = dict(zip(heads,vals))
    minikey = [v for k, v in sorted(obj.items(), key=lambda x:x[0]) if k in keys ]  
    minikey = '_'.join( minikey )
    val = key_val[ minikey ] 
    #print(key, minikey, val)
    
    text = line.strip() + f',{val}'
    ft.write( text + '\n' )

  # input_nameを更新
  input_name = f'{input_name}c'
