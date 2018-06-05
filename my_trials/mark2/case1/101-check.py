
import pandas as pd


dft = pd.read_csv('sub_it.csv')
dfs = pd.read_csv('../../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/sample_submission.csv')

print( all((dft['click_id'] == dfs['click_id']).tolist()) )

