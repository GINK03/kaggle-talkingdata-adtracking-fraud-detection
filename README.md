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

## case10 
new baseline archived!

## case11
add label encoding...
 対照実験 :-> あると0.9721と下がりまくった。原因はTEかどうかチェックを行う (原因だった場合、取り除く)
 
 実験結果 n=1 -> 取り除いたところ0.9782と微妙に回復した

## case12
#### 対象実験
x1, x7, x4, nextClick_shift, dayを除いて実験
 
#### 対照実験結果
0.9782 -> 0.9785
kernel -> funを編集して対応

#### 実験対象
ip_chl_indexのカテゴリ変数を与えてみる
#### 対照実験結果
0.9785　-> 0.9779
改悪のため取り下げ(test_a.py)

ip-chl, os-chl分散エンコーディング

## case13 
#### 対照実験
epoch timeを追加
#### 実験結果
0.9785　-> 0.9785（変化なし）

#### 対照実験
['ip', 'device', 'os', 'app', 'channel']でchannelで数をカウント、['ip', 'device', 'os', 'app', 'channel']でchannelで平均をカウント,['ip', 'device', 'os', 'app', 'channel']でchannelで分散をカウント
