import os
import sys
fp_result = open('./submission_auc=0.986345537189_windows=000000000000_est=800_6d610de7a06537c85cfaf47c853927a933e62fda6f9fc5109a16a71d3e5f6b9d.csv')

fp_prepare = open('./files/test_df_000010000000.csv')
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
  
