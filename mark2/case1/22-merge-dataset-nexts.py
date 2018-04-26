
import pandas as pd
import sys

if '1' in sys.argv:
  dfn = pd.read_csv('var/all_nextclick.csv')
  dft = pd.read_csv('var/train.csv')

  df = dft.set_index('control_index').join( dfn.set_index('control_index') )

  df.to_csv('var/train_nexts.csv')
