
import pandas as pd
import sys
import gc

import concurrent.futures
import pickle
import gzip
def _map(arg):
  index, df = arg
  print(index)
  
  key_atrs = {}
  key_atrs2 = {}
  for obj in df.to_dict(orient='records'):  
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
 
  data = pickle.dumps( (key_atrs, key_atrs2) )
  data = gzip.compress( data )
  open(f'files/tmp/middle_chunk_{index:09d}.pkl.gz', 'wb').write( data )

  print('finish', index)

def mapper(data):
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map(_map, data)

if '--encode' in sys.argv:
  if '--init' in sys.argv:
    sample_dfs = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,1), nrows=14000_0000, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], chunksize=100000)
    data = []
    for index, sample_df in enumerate(sample_dfs):
      data.append( (index, sample_df) )
      if len(data) >= 16:
        mapper(data)
        data = []
    mapper(data)
    sys.exit()
  
  from pathlib import Path
  if '--reduce' in sys.argv:
    key_atrs1 = {}
    key_atrs2 = {}
    for path in Path('./files/tmp').glob('middle_chunk_*'):
      _key_atrs1, _key_atrs2 = pickle.loads(gzip.decompress(path.open('rb').read()))
      for key, atrs in _key_atrs1.items():
        if key_atrs1.get(key) is None:
          key_atrs1[key] = [0, 0]
        key_atrs1[key][0] += atrs[0]
        key_atrs1[key][1] += atrs[1]
      for key, atrs in _key_atrs2.items():
        if key_atrs2.get(key) is None:
          key_atrs2[key] = [0, 0]
        key_atrs2[key][0] += atrs[0]
        key_atrs2[key][1] += atrs[1]

    buff = []
    for key, atrs in key_atrs1.items():
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


if '--merge' in sys.argv:
  for gp_file, join_key in [('./files/appchl_df.pkl', ['app', 'channel']), ('./files/oschl_df.pkl', ['os', 'channel'])]:
    sample_df = pd.read_pickle(gp_file)
    for name in ['train', 'val', 'test']:
      train_df = pd.read_pickle(f'./files/{name}_df.pkl') 
      train_df = train_df.merge(sample_df, on=join_key, how='left')
      train_df = train_df.fillna(-1.0)
      train_df.to_pickle(f'./files/{name}_df.pkl')
    del train_df; gc.collect()
