
window = 0

fp_train = f'./files/train_df_dhf_hm_{window:012d}.csv'
fp_val   = f'./files/val_df_dhf_hm_{window:012d}.csv'
fp_test  = f'./files/test_df_dhf_hm_{window:012d}.csv'

import csv
from datetime import datetime
import pickle
import gzip
import sys
if '--pickle' in sys.argv:
  channelip_dh_f = {}
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
      channel = obj['channel']
      ip      = obj['ip']
      app     = obj['app']
      channelip = f'{channel}_{ip}_{app}'
      dt = datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S')
      dh = (dt.hour*60 + dt.minute)//10
      if channelip_dh_f.get(channelip) is None:
        channelip_dh_f[channelip] = {}
      if channelip_dh_f[channelip].get(dh) is None:
        channelip_dh_f[channelip][dh] = 0
      channelip_dh_f[channelip][dh] += 1

    for channelip, dh_f in channelip_dh_f.items():
      print(channelip, dh_f)
  data = gzip.compress( pickle.dumps( channelip_dh_f ) )
  open('files/tmp/channelip_dh_f.pkl.gz', 'wb').write( data )

if '--convert' in sys.argv:
  data = open('files/tmp/channelip_dh_f.pkl.gz', 'rb').read()
  ip_dh_f = pickle.loads( gzip.decompress( data ) )

  for (name, filename) in [ \
                            (f'val_df_dhf_hm_ci_{window:012d}.csv', fp_val), \
                            (f'test_df_dhf_hm_ci_{window:012d}.csv', fp_test), \
                            (f'train_df_dhf_hm_ci_{window:012d}.csv', fp_train)
                          ]:
    # 書き出し 
    ''' header name -> ci '''
    fp = open(filename)
    fp_w = open(f'files/{name}', 'w')
    it = csv.reader( fp )
    head = next(it)
    head.append('ci') 
    text = ','.join(head)
    fp_w.write( f'{text}\n' )
    
    for vals in it:
      obj     = dict(zip(head,vals)) 
      strtime = obj['click_time']
      ip      = obj['ip']
      channel = obj['channel']
      app     = obj['app']
      cip = f'{channel}_{ip}_{app}'
      dt      = datetime.strptime(strtime, '%Y-%m-%d %H:%M:%S')
      dh = (dt.hour*60 + dt.minute)//10
      dh_f    = ip_dh_f[cip]
      f       = dh_f[dh]
      vals.append( f'{f}' ) 
      text = ','.join(vals)
      fp_w.write( f'{text}\n' )
