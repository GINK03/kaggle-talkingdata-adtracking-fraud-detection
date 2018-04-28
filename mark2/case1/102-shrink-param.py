import os

import csv


import json
import sys

filt = sorted( json.load(open('files/base20180428')), key=lambda x:x[1] )[:-20]
print(filt)
filt = set( [ x[0] for x in filt ] )
print(filt)

if 'train' in sys.argv:
  fp = open('var/train_nexts.csv')
  it = csv.DictReader(fp)
  key_ft = {}
  for index, obj in enumerate(it):
    key = index//100000
    for x in list(obj.keys()):
      if x in filt:
        del obj[x]
    # click timeも消す
    for x in ['click_time']:
      del obj[x]
    if key_ft.get(key) is None:
      for _, ft in key_ft.items():
        del ft 
      key_ft[key] = csv.DictWriter(open(f'var/shrink_chunk/shrink_train_nexts_{key:09d}.csv', 'w'), fieldnames=list(obj.keys()))
      key_ft[key].writeheader()
    key_ft[key].writerow( obj )
    if index%10000 == 0:
      print(f'now scan {index}')

if 'test' in sys.argv:
  fp = open('var/test_nexts.csv')
  it = csv.DictReader(fp)
  key_ft = {}
  for index, obj in enumerate(it):
    key = index//100000
    for x in list(obj.keys()):
      if x in filt:
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
