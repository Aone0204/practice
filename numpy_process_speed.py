#numpyを使う場合と使わない場合の処理速度を比較する。
#numpyを用いて配列 mat の平均を、1番目の軸に沿って計算する。

 
#  必要なライブラリをimport
import numpy as np
import time
from numpy.random import rand

# 行、列の大きさ
N = 5000

# 配列の初期化
mat = rand(N, N)

# Numpyの機能を使わずに計算

# 開始時間の取得
start = time.time()

# for文を使って、1番目の軸に沿って平均を計算
mean_not_numpy = []
for i in range(N):
    mean_not_numpy.append(sum(mat[i]) / len(mat[i]))

# 出力形式を整えるため、numpy配列に変換
print(np.array(mean_not_numpy))
print(f'Total time when not using NumPy：{(time.time() - start):.2f}[sec]')
print()

# NumPyを使って計算

# 開始時間の取得
start = time.time()

# NumPyの機能を使って、1番目の軸に沿って平均を計算
mean_numpy = mat.mean(axis=1)

print(mean_numpy)
print(f'Total time when using NumPy：{(time.time() - start):.2f}[sec]')