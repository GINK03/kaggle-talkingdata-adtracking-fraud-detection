
import pandas as pd
import sys
import gc

if '--encode' in sys.argv:
  sample_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,7000_0000), nrows=4000_0000,usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
  sample_df = sample_df.infer_objects()

  key_atrs = {}
  key_atrs2 = {}
  for obj in sample_df.to_dict(orient='records'):  
    app = obj['app']
    chl = obj['channel']
    os = obj['os']
    key = f'{app}_{chl}'
    if key_atrs.get(key) is None:
      key_atrs[key] = [0, 0]
    is_attributed = obj['is_attributed']
    key_atrs[key][0] += is_attributed
    key_atrs[key][1] += 1
    
    key = f'{os}_{chl}'
    if key_atrs2.get(key) is None:
      key_atrs2[key] = [0, 0]
    is_attributed = obj['is_attributed']
    key_atrs2[key][0] += is_attributed
    key_atrs2[key][1] += 1
    
  buff = []
  for key, atrs in key_atrs.items():
    mean = atrs[0]/atrs[1]
    app, chl = key.split('_')
    buff.append( {'app':int(app), 'channel':int(chl), 'app_chl_conf':mean} )
  gp = pd.DataFrame(buff)
  del buff
  gp.info()
  print( 'gp', gp.head() )
  gp.to_pickle(f'files/appchl_df.pkl')

  buff = []
  for key, atrs in key_atrs2.items():
    mean = atrs[0]/atrs[1]
    os, chl = key.split('_')
    buff.append( {'os':int(os), 'channel':int(chl), 'os_chl_conf':mean} )
  gp2 = pd.DataFrame(buff)
  del buff
  gp2.info()
  print( 'gp2', gp2.head() )
  gp2.to_pickle(f'files/oschl_df.pkl')


  train_df = sample_df.merge(gp, on=['app', 'channel'], how='left')
  train_df = train_df.merge(gp2, on=['os', 'channel'], how='left')
  train_df = train_df.fillna(-1.0)
  print( 'train', train_df.head() )

if '--merge' in sys.argv:
  for gp_file, join_key in [('./files/appchl_df.pkl', ['app', 'channel']), ('./files/oschl_df.pkl', ['os', 'channel'])]:
    sample_df = pd.read_pickle(gp_file)
    for name in ['train', 'val', 'test']:
      train_df = pd.read_pickle(f'./files/{name}_df.pkl') 
      train_df = train_df.merge(sample_df, on=join_key, how='left')
      train_df.fillna(-1.0)
      train_df.to_pickle('./file/{name}_df.pkl')
    del train_df; gc.collect()
