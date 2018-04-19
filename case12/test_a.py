import pandas as pd


df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/train.csv')
df.append( df_test )
df['factrize'] = df['ip'].astype(str) + "_" + df['channel'].astype(str)
buff = []
for index, factrize in enumerate( set( df['factrize'].tolist() ) ):
  obj = {'ip_chl_ind':index, 'factrize': factrize}
  buff.append( obj )
key_df = pd.DataFrame( buff )
df = df.merge(key_df, on=['factrize'], how='left')
df.info()
print(df.head())
