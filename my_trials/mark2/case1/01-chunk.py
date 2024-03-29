import datetime
import itertools
import sys
fp = open('../../input/train.csv')
key_fp = {}

head = next(fp).strip()
open('var/head', 'w').write( 'control_index,'+ head +',wday,day,hour,minute')
control = 0
for index, line in enumerate(fp):
  if index%10000 == 0:
    print(f'now index {index}')
  key = control//500000
  line = line.strip()
  raw = line.split(',')[5]
  dt = datetime.datetime.strptime(raw, '%Y-%m-%d %H:%M:%S')
  times = ','.join( [ f'{x}' for x in [ dt.weekday(), dt.day, dt.hour, dt.minute ] ] )
  text = f'{control:012d},' + line + f',{times}'
  if key_fp.get(key) is None:
    key_fp[key] = open(f'var/chunks/{key:09d}', 'w')
  key_fp[key].write( text + '\n')
  control += 1

fp2 = open('../../input/test.csv')
head = next(fp2).strip()
open('var/head_test', 'w').write( 'control_index,'+ head +',wday,day,hour,minute')
for index, line in enumerate(fp2):
  if index%10000 == 0:
    print(f'now test index {index}')
  key  = f'test_{control//500000:09d}'
  line = line.strip()
  raw = line.split(',')[6]
  dt = datetime.datetime.strptime(raw, '%Y-%m-%d %H:%M:%S')
  times = ','.join( [ f'{x}' for x in [ dt.weekday(), dt.day, dt.hour, dt.minute ] ] )
  text = f'{control:012d},' + line + f',{times}'
  if key_fp.get(key) is None:
    key_fp[key] = open(f'var/chunks/test_{key}', 'w')
  key_fp[key].write( text + '\n')
  control += 1
