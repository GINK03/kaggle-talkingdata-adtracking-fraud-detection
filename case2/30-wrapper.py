
import os

import glob

import random

import sys
confs = glob.glob('./files/confs/*')
random.shuffle(confs)

conf = confs.pop()
os.system(f'lightgbm config={conf}') 
sys.exit() 
result = os.popen(f'lightgbm config={conf}').read()
sys.exit() 

