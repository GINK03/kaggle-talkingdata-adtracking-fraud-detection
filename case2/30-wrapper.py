
import os

import glob

import random

import sys
 
confs = glob.glob('./files/confs/*')

random.shuffle(confs)

conf = confs.pop()

ha = conf.split('/').pop()
os.system(f'lightgbm config={conf}') 
sys.exit() 

result = os.popen(f'lightgbm config={conf}').read()
open(f'./files/results/{ha}', 'w').write( result )

sys.exit() 

