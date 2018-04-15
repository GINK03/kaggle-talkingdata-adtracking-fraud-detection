
window = 0

fp_train = f'./files/train_df_{window:012d}.csv'
fp_val   = f'./files/val_df_{window:012d}.csv'
fp_test  = f'./files/test_df_{window:012d}.csv'

import csv
from datetime import datetime
import pickle
import gzip
import sys
if '--pickle' in sys.argv:
  ip_dh_f = {}
  for (name, filename) in [ \
                            (f'val_df_dhf_{window:012d}.csv', fp_val), \
                            (f'test_df_dhf_{window:012d}.csv', fp_test), \
                            (f'train_df_dhf_{window:012d}.csv', fp_train)
                          ]:
    fp = open(filename)
    it = csv.reader( fp )
    head = next(it)

    for vals in it:
      obj = dict(zip(head,vals)) 
      strtime = obj['click_time']
      ip      = obj['ip']
      dt = datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S')
      dh = f'{dt.day:02d}_{dt.hour:02d}'
      if ip_dh_f.get(ip) is None:
        ip_dh_f[ip] = {}
      if ip_dh_f[ip].get(dh) is None:
        ip_dh_f[ip][dh] = -1
      ip_dh_f[ip][dh] += 1

    for ip, dh_f in ip_dh_f.items():
      print(ip, dh_f)
  data = gzip.compress( pickle.dumps( ip_dh_f ) )
  open('files/tmp/ip_dh_f.pkl.gz', 'wb').write( data )

if '--convert' in sys.argv:
  data = open('files/tmp/ip_dh_f.pkl.gz', 'rb').read()
  ip_dh_f = pickle.loads( gzip.decompress( data ) )

  for (name, filename) in [ \
                            (f'val_df_dhf_{window:012d}.csv', fp_val), \
                            (f'test_df_dhf_{window:012d}.csv', fp_test), \
                            (f'train_df_dhf_{window:012d}.csv', fp_train)
                          ]:
    # 書き出し 
    ''' header name -> dh_f '''
    fp = open(filename)
    fp_w = open(f'files/{name}', 'w')
    it = csv.reader( fp )
    head = next(it)
    head.append('dh_f') 
    text = ','.join(head)
    fp_w.write( f'{text}\n' )
    
    for vals in it:
      obj     = dict(zip(head,vals)) 
      strtime = obj['click_time']
      ip      = obj['ip']
      dt      = datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S')
      dh      = f'{dt.day:02d}_{dt.hour:02d}'
      dh_f    = ip_dh_f[ip]
      f       = dh_f[dh]
      vals.append( f'{f}' ) 
      text = ','.join(vals)
      fp_w.write( f'{text}\n' )
