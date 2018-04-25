
from pathlib import Path

import gzip, pickle

import os

import concurrent.futures

keys = []
for name in Path('./var/03').glob('*'):
  key = str(name).split('/').pop()
  if os.path.exists(f'var/{key}_all.pkl.gz'):
    continue
  keys.append( key )

def pmap(key):
  print(key)
  p_freq = {}
  for name in sorted(Path(f'var/03/{key}').glob('*')):
    obj = pickle.loads(gzip.decompress(name.open('rb').read()))
    #kprint(key, obj)
    for p, space in obj.items():
      if p_freq.get(p) is None:
        p_freq[p] = set()
      p_freq[p].union(space)

  for p in p_freq.keys():
    p_freq[p] = len(p_freq)
  print( key, len(p_freq) )
  d = gzip.compress(pickle.dumps(p_freq))
  open(f'var/{key}_all.pkl.gz', 'wb').write( d )

with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
  exe.map(pmap, keys)
