
from pathlib import Path
import statistics
import random
names = sorted([name for name in Path('./').glob('submission*.csv')])

names = [name for name in names[:-5]]

fps = [name.open() for name in names]

heads = [next(fp).strip() for fp in fps]
head = heads[0]

fp = open('ensemble.csv', 'w')
fp.write( f'{head}\n' )
while True:
  try:
    vals = [next(fp).strip() for fp in fps]
  except Exception as ex:
    print(ex)
    break
  index = vals[0].split(',').pop(0)
  mean = statistics.mean( [float(val.split(',').pop()) for val in vals ] )
  fp.write( f'{index},{mean}\n' )
  if random.random() < 0.001:
    print(index, mean)
