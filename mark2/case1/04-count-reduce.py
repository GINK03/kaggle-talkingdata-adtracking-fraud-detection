
from pathlib import Path

import gzip, pickle

import os
keys = []
for name in Path('./var/02').glob('*'):
  key = str(name).split('/').pop()
  if 'proceed' in key:
    continue
  if os.path.exists(f'var/{key}_count_all.pkl.gz'):
    continue
  keys.append( key )

import concurrent.futures


def pmap(key):
  print(key)
  p_freq = {}
  for name in sorted(Path(f'var/02/{key}').glob('*')):
    obj = pickle.loads(gzip.decompress(name.open('rb').read()))
    #kprint(key, obj)
    for p, freq in obj.items():
      if p_freq.get(p) is None:
        p_freq[p] = 0
      p_freq[p] += freq
  d = gzip.compress(pickle.dumps(p_freq))
  open(f'var/{key}_count_all.pkl.gz', 'wb').write( d )

with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
  exe.map(pmap, keys)
