
import os

import json

import sys

if '--step1' in sys.argv:
  fp = open('../../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/mnt/ssd/kaggle-talkingdata2/competition_files/train.csv')

  head = next(fp).strip().split(',') 

  seq_fp = {}
  for index, line in enumerate(fp):
    seq = index//1000_000
    if seq_fp.get(seq) is None:
      print(f'now seq {seq}')
      seq_fp[seq] = open(f'files/data/train_{seq:012d}.json', 'w')
    val = line.strip().split(',')
    obj = dict( zip(head, val) )
    data = json.dumps(obj)
    seq_fp[seq].write( f'{data}\n' )

if '--step2' in sys.argv:
  fp = open('../../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/test.csv')

  head = next(fp).strip().split(',') 

  seq_fp = {}
  for index, line in enumerate(fp):
    seq = index//1000_000
    if seq_fp.get(seq) is None:
      print(f'now seq {seq}')
      seq_fp[seq] = open(f'files/data/test_{seq:012d}.json', 'w')
    val = line.strip().split(',')
    obj = dict( zip(head, val) )
    data = json.dumps(obj)
    seq_fp[seq].write( f'{data}\n' )
    
