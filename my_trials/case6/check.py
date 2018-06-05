import os
import sys
fp_result = open('./submission_0.985746069639_000000000000_418250405c9c198adb3d2ebac6af50317a97cff0e867edec5ad0022ad3ff34b1.csv')

fp_prepare = open('./files/test_df_000000000000.csv')
fp_sample = open('../../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/sample_submission.csv')


# indexのチェック
if '--index' in sys.argv:
  for (res, sam, pre) in zip(fp_result, fp_sample, fp_prepare):
    res, sam, pre = map(lambda x:x.strip(), [res, sam, pre])
    res_index = res.split(',').pop(0)
    sam_index = sam.split(',').pop(0)
    pre_index = sam.split(',').pop(0)
    if pre_index != sam_index:
      print(res, sam)

if '--double' in sys.argv:
  # indexが再定義されると書いてあって、？となっているエラーのトレース
  res_indexs = set()

  index_posi = {} 
  for posi, res in enumerate(fp_result):
    res_index = res.split(',').pop(0)
    
    if res_index not in res_indexs:
      res_indexs.add( res_index )
      index_posi[ res_index ] = posi
    else:
      print( f'double count old_posi={index_posi[res_index]} posi={posi}, res_index={res_index}')
  
