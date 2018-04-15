# 

## case1
試行錯誤のケース  
- データ数が多すぎで、安定して運用できる方法を探した
- 0.8992
- RF-benchに負ける

## case2
分散処理を大幅に取り入れて諸々高速化
- 特徴量探索を早めた

## case3 
- case2をベースに時間の日にちの特徴量を追加
- impの大きさのカテゴリ変数を削除

## case4 
- Kernelの特徴量をつかってベースラインを構築する
- 3つのアンサンブルを利用してRLでfit
- データセットを10分割

## case5 
- kernelの手法に一部巻き戻し
- 機体性能OK
- 特徴量追加後、kernel最高峰程度を目指す　

## case6
- range探索(どこまで伸ばせはうれしいか？)
- いい感じに安定したのでアンカーポイントにする

## case7 
- ipのウィンドウ周波数関数を実装して, jsonで吐き出し(middle) -> pandasで突合する
- pub scoreが0.9695で変わらず -> 残念
-　日付指示ベクトルを追加予定
- middle-ip-dt.py -> あまり改善せず
-　　middle-date.py
- まず、リニアでチェック　
- feature_importanceを利用するようになった -> channel\*ip freqを入れたいモチベになった
```console
01: middle-ip-dt.py
02: middle-date.py
03: middle-channle_ip-dt.py 
```
