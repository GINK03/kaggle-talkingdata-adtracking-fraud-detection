import csv

it = csv.DictReader(open('var/test.csv'))
it2 = csv.DictReader(open('../../../.kaggle/competitions/talkingdata-adtracking-fraud-detection/sample_submission.csv'))
import random
while True:
  o1, o2 = next(it), next(it2)
  cid1 = o1['click_id']
  cid2 = o2['click_id']
  if random.random() < 0.0001:
    print(cid1, cid2)
