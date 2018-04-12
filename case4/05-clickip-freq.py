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
    ip_freq = {}
    app_freq = {}
    ipXapp_freq = {}
    ipXos_freq = {}
    ipXappXos_freq = {}
    for index, line in enumerate(path.open()):
      if index%100000 == 0:
        print(f'now iter {index}@{key}/{path}')
      obj = json.loads(line.strip())
      ip  = obj['ip']
      app  = obj['app']
      os = obj['os']

      ipXapp = f'{ip}_x_{app}'
      ipXos = f'{ip}_x_{os}'
      ipXappXos = f'{ip}_x_{app}_x_{os}'
      if ip_freq.get(ip) is None: 
        ip_freq[ip] = 0
      ip_freq[ip] += 1
      if app_freq.get(app) is None: 
        app_freq[app] = 0
      app_freq[app] += 1
      if ipXapp_freq.get(ipXapp) is None:
        ipXapp_freq[ipXapp] = 0
      ipXapp_freq[ipXapp] += 1
      if ipXos_freq.get(ipXos) is None:
        ipXos_freq[ipXos] = 0
      ipXos_freq[ipXos] += 1
      if ipXappXos_freq.get(ipXappXos) is None:
        ipXappXos_freq[ipXappXos] = 0
      ipXappXos_freq[ipXappXos] += 1
      

    return (ip_freq, app_freq, ipXapp_freq, ipXos_freq, ipXappXos_freq)

  ip_freq, app_freq, ipXapp_freq, ipXos_freq, ipXappXos_freq = {}, {}, {}, {}, {}
  args = [(index, path) for index, path in enumerate(sorted(Path('./files/data').glob('train_*')))]
  [args.append( (index,path) ) for index, path in enumerate(sorted(Path('./files/data').glob('test_*')))]
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    for (_ip_freq, _app_freq, _ipXapp_freq, _ipXos_freq, _ipXappXos_freq) in exe.map(_map_train, args):
      for ip, freq in _ip_freq.items():
        if ip_freq.get(ip) is None:
          ip_freq[ip] = 0
        ip_freq[ip] += freq
      for app, freq in _app_freq.items():
        if app_freq.get(app) is None:
          app_freq[app] = 0
        app_freq[app] += freq
      for ipXapp, freq in _ipXapp_freq.items():
        if ipXapp_freq.get(ipXapp) is None:
          ipXapp_freq[ipXapp] = 0
        ipXapp_freq[ipXapp] += freq
      for ipXos, freq in _ipXos_freq.items():
        if ipXos_freq.get(ipXos) is None:
          ipXos_freq[ipXos] = 0
        ipXos_freq[ipXos] += freq
      for ipXappXos, freq in _ipXappXos_freq.items():
        if ipXappXos_freq.get(ipXappXos) is None:
          ipXappXos_freq[ipXappXos] = 0
        ipXappXos_freq[ipXappXos] += freq

  json.dump([ip_freq, app_freq, ipXapp_freq, ipXos_freq, ipXappXos_freq],  fp=open('files/freqs.json', 'w'), indent=2)

