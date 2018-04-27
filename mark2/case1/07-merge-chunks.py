
from pathlib import Path

import pickle, gzip

import glob

import sys
# ヘッダー作成
apps = [x.replace('.pkl.gz', '') for x in sum([sorted(glob.glob('var/*_count_all.pkl.gz')), sorted(glob.glob('var/*_uniq_uniq_all.pkl.gz'))], [])]
apps = ','.join(apps)
print(apps)
if 'train' in sys.argv:
  head = open('var/head').read() + ',' + apps
  fp = open('var/train.csv', 'w')
  fp.write( head + '\n' )
  for name in sorted( filter(lambda x:'test_' not in x, sorted(glob.glob('var/chunks/*_finished'))) ):
    print(name)
    for line in open(name):
      fp.write( line )

if 'test' in sys.argv:
  head = open('var/head_test').read() + ',' + apps
  fp = open('var/test.csv', 'w')
  fp.write( head + '\n' )
  for name in sorted( filter(lambda x:'test_' in x, sorted(glob.glob('var/chunks/*_finished'))) ):
    print(name)
    for line in open(name):
      fp.write( line )
