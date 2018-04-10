import os

import sys

from pathlib import Path

import json
  
import numpy as np

from scipy import sparse

import pickle

import gzip
if '--step1' in sys.argv: # feat indexを作成
  fp = open('../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/mnt/ssd/kaggle-talkingdata2/competition_files/train.csv')

  head = next(fp).strip().split(',')

  click_ip_freq  = {}
  nclick_ip_freq = {}
  for index, line in enumerate(fp):
    if index%100000 == 0:
      print(f'now iter {index}')
    val = line.strip().split(',')
    obj = dict( zip(head, val) )
    is_attributed = obj['is_attributed']
    ip  = obj['ip']
    if is_attributed == '1':
      if click_ip_freq.get(ip) is None: 
        click_ip_freq[ip] = 0
      click_ip_freq[ip] += 1
    else:
      if nclick_ip_freq.get(ip) is None: 
        nclick_ip_freq[ip] = 0
      nclick_ip_freq[ip] += 1


  json.dump(click_ip_freq,  fp=open('files/click_ip_freq.json', 'w'), indent=2)
  json.dump(nclick_ip_freq,  fp=open('files/nclick_ip_freq.json', 'w'), indent=2)

if '--step2' in sys.argv:
  # logを取って、5分割とか
  ...
  

