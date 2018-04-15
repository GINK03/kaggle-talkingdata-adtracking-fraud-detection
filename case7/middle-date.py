
window = 0

fp_train = f'./files/train_df_dhf_{window:012d}.csv'
fp_val   = f'./files/val_df_dhf_{window:012d}.csv'
fp_test  = f'./files/test_df_dhf_{window:012d}.csv'

import csv
from datetime import datetime
import pickle
import gzip
import sys

if '--pickle' in sys.argv:
  hms = set()
  for (name, filename) in [ \
                            (f'val_df_dhf_hm_{window:012d}.csv', fp_val), \
                            (f'test_df_dhf_hm_{window:012d}.csv', fp_test), \
                            (f'train_df_dhf_hm_{window:012d}.csv', fp_train)
                          ]:
    fp = open(filename)
    it = csv.reader( fp )
    head = next(it)

    for vals in it:
      obj = dict(zip(head,vals)) 
      strtime = obj['click_time']
      ip      = obj['ip']
      dt = datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S')
      hm = (dt.hour * 60 + dt.minute)//10
      hms.add( int(hm) ) 

  hm_index = {}
  for index, hm in enumerate(sorted(hms)):
    print( hm, index )
    hm_index[hm] = index

  data = gzip.compress( pickle.dumps( hm_index ) )
  open('files/tmp/hm_index.pkl.gz', 'wb').write( data )

if '--convert' in sys.argv:
  data = open('files/tmp/hm_index.pkl.gz', 'rb').read()
  hm_index = pickle.loads( gzip.decompress( data ) )
  for (name, filename) in [ \
                            (f'val_df_dhf_hm_{window:012d}.csv', fp_val), \
                            (f'test_df_dhf_hm_{window:012d}.csv', fp_test), \
                            (f'train_df_dhf_hm_{window:012d}.csv', fp_train)
                          ]:
    # 書き出し 
    ''' header name -> hm '''
    fp = open(filename)
    fp_w = open(f'files/{name}', 'w')
    it = csv.reader( fp )
    head = next(it)
    head.append('hm') 
    text = ','.join(head)
    fp_w.write( f'{text}\n' )
    
    for vals in it:
      obj = dict(zip(head,vals)) 
      strtime = obj['click_time']
      ip      = obj['ip']
      dt = datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S')
      hm = (dt.hour * 60 + dt.minute)//10
      index = hm_index[ hm ]
      vals.append( f'{index}' ) 
      text = ','.join(vals)
      fp_w.write( f'{text}\n' )
