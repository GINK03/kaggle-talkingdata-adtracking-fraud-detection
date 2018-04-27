
import os
import glob

# 間違ってたら手動で削除して
for name in glob.glob('var/*_nextclick.csv'):
  count = os.popen(f'cat {name} | wc -l ').read().strip()
  print(name, count)

fps = [open(name) for name in sum([sorted(glob.glob('var/*_nextclick.csv')), sorted(glob.glob('var/*_nextnextclick.csv')), sorted(glob.glob('var/*_prev.csv'))], [])]

control = 0
head = 'control_index,' + ','.join( [name.split('/').pop().replace('.csv', '') for name in \
  sum([sorted(glob.glob('var/*_nextclick.csv')), sorted(glob.glob('var/*_nextnextclick.csv')), sorted(glob.glob('var/*_prev.csv'))], [])] )

open('var/head_nextclicks', 'w').write( head )
ft = open(f'var/all_nextclick.csv', 'w')
ft.write( head + '\n' )
while True:
  try:
    text = ','.join( [next(fp).strip() for fp in fps] )
  except Exception as ex:
    print(ex)
    print('finish')
    break
  ft.write( f'{control},' + text + '\n' )
  control += 1
  if control % 10000 == 0:
    print( f'now {control}' )
  #print(text) 
   
