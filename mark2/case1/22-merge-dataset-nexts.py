
import pandas as pd
import sys

if '1' in sys.argv:
  dfn = pd.read_csv('var/all_nextclick.csv')
  dft = pd.read_csv('var/train.csv')

  df = dft.set_index('control_index').join( dfn.set_index('control_index') )
  df.to_csv('var/train_nexts.csv')

if '2' in sys.argv:
  dfn = pd.read_csv('var/all_nextclick.csv')
  dft = pd.read_csv('var/train.csv')

  df = dft.set_index('control_index').join( dfn.set_index('control_index') )
  df.to_csv('var/train_nexts.csv')

if '1-1' in sys.argv:
  
  import csv
  import pickle
  import os

  if not os.path.exists('var/all_nextclick_cindex_val.pkl'):
    cindex_val = {}
    f = open('var/all_nextclick.csv')
    for ind, obj in enumerate(csv.DictReader(f)):
      if ind%10000 == 0:
        print(f'now {ind}')
      cindex = int( obj['control_index'] )
      cindex_val[cindex] = ','.join( obj.values() )
    data = pickle.dumps(cindex_val)
    open('var/all_nextclick_cindex_val.pkl', 'wb').write( data )

  cindex_val = pickle.load( open('var/all_nextclick_cindex_val.pkl', 'rb') )
  f = open('var/train.csv')
  ft = open('var/train_nexts.csv', 'w')
  head = next(f).strip() + ',' + next( open('var/all_nextclick.csv') ).strip()
  ft.write( head + '\n' )
  for line in f:
    line = line.strip()
    try: 
      cindex = int( line.split(',').pop(0) )
    except Exception as ex:
      print(ex)
      continue
    text = line + ',' + cindex_val[cindex] 
    ft.write( text + '\n' )

  f = open('var/test.csv')
  ft = open('var/test_nexts.csv', 'w')
  head = next(f).strip() + ',' + next(f).strip()
  ft.write( head + '\n' )
  for line in f:
    line = line.strip()
    try: 
      cindex = int( line.split(',').pop(0) )
    except Exception as ex:
      print(ex)
      continue
    text = line + ',' + cindex_val[cindex] 
    ft.write( text + '\n' )


