
import os

import glob

import random

import sys

from pathlib import Path

confs = glob.glob('./files/confs/*')

random.shuffle(confs)

for conf in confs:
  ha = conf.split('/').pop()
  if Path(f'./files/results/{ha}').exists():
    continue
  result = os.popen(f'lightgbm config={conf}').read()
  open(f'./files/results/{ha}', 'w').write( result )

