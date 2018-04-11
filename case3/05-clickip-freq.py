import os

import sys

from pathlib import Path

import json
  
import numpy as np

from scipy import sparse

import pickle

import gzip

import concurrent.futures
if '--step1' in sys.argv: # feat indexを作成
  
  # train data
  def _map_train(arg):
    key, path = arg
    click_ip_freq = {}
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now iter {index}@{key}/{path}')
      obj = json.loads(line.strip())
      ip  = obj['ip']
      if click_ip_freq.get(ip) is None: 
        click_ip_freq[ip] = 0
      click_ip_freq[ip] += 1
    return click_ip_freq

  click_ip_freq = {}
  args = [(index, path) for index, path in enumerate(Path('./files/data').glob('train_*'))]
  [args.append( (index,path) ) for index, path in enumerate(Path('./files/data').glob('test_*'))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    for _click_ip_freq in exe.map(_map_train, args):
      for ip, freq in _click_ip_freq.items():
        if click_ip_freq.get(ip) is None:
          click_ip_freq[ip] = 0
        click_ip_freq[ip] += freq

  json.dump(click_ip_freq,  fp=open('files/click_ip_freq.json', 'w'), indent=2)

