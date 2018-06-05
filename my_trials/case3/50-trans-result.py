fp = open('./LightGBM_predict_result.txt')

fw = open('./files/submission.csv', 'w')
fw.write( 'click_id,is_attributed\n' )

for index, line in enumerate(fp):
  fw.write( f'{index},{line.strip()}\n')
