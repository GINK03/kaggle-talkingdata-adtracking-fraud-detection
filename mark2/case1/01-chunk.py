
fp = open('../../input/train.csv')

key_fp = {}
for index, line in enumerate(fp):
  
  if index%10000 == 0:
    print(f'now index {index}')
  key = index//100000
  line = line.strip()
  text = f'{index:012d}' + line
  
  if key_fp.get(key) is None:
    key_fp[key] = open(f'var/chunks/{key:09d}', 'w')
  
  key_fp[key].write( text + '\n')
  
