import os

import csv


import json
import sys

#filt = sorted( json.load(open('files/base20180428')), key=lambda x:x[1] )[:-20]
#print(filt)
#filt = set( [ x[0] for x in filt ] )
#print(filt)
use = 'ip,app,device,os,channel,click_time,attributed_time,is_attributed,hour,var/app_ip_os_count_all,var/app_ip_wday_count_all,var/app_wday_count_all,var/device_hour_ip_count_all,var/device_ip_count_all,var/device_ip_os_count_all,var/device_ip_wday_count_all,var/hour_ip_count_all,var/hour_ip_wday_count_all,var/ip_os_wday_count_all,var/ip_wday_count_all,ip_os_app_device_nextclick,ip_os_app_nextclick,ip_os_app_device_nextnextclick,ip_os_app_device_prevclick'.split(',')

if 'train' in sys.argv:
  fp = open('var/train_nexts.csv')
  it = csv.DictReader(fp)
  key_ft = {}
  for index, obj in enumerate(it):
    key = index//100000
    for x in list(obj.keys()):
      if x not in use: 
        del obj[x]
    # click timeも消す
    for x in ['click_time']:
      del obj[x]
    if key_ft.get(key) is None:
      for _, ft in key_ft.items():
        ft[1].close()
        del ft 
      _fp = open(f'var/shrink_chunk/shrink_train_nexts_{key:09d}.csv', 'w')
      _dict_writer = csv.DictWriter(_fp, fieldnames=list(obj.keys()))
      key_ft[key] = [_dict_writer, _fp] 
      key_ft[key][0].writeheader()
    key_ft[key][0].writerow( obj )
    if index%10000 == 0:
      print(f'now scan {index}')

if 'test' in sys.argv:
  fp = open('var/test_nexts.csv')
  it = csv.DictReader(fp)
  key_ft = {}
  for index, obj in enumerate(it):
    key = index//100000
    for x in list(obj.keys()):
      if x not in use:
        del obj[x]
    # click timeも消す
    for x in ['click_time']:
      del obj[x]
    
    if key_ft.get(key) is None:
      for _, ft in key_ft.items():
        del ft 
      key_ft[key] = csv.DictWriter(open(f'var/shrink_chunk/shrink_test_nexts_{key:09d}.csv', 'w'), fieldnames=list(obj.keys()))
      key_ft[key].writeheader()
    key_ft[key].writerow( obj )
    if index%10000 == 0:
      print(f'now scan {index}')
