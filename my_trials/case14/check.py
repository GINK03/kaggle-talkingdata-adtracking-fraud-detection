import pandas as pd


sample_df = pd.read_csv('~/.kaggle/competitions/talkingdata-adtracking-fraud-detection/sample_submission.csv')

import glob

for name in glob.glob('./*.csv'):
  target_df = pd.read_csv(name)

  comp = sample_df[ ['click_id'] ] == target_df[ ['click_id'] ]
  
  print( all(comp['click_id'].tolist()), name )
