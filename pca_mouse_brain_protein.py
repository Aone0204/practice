#マウスの大脳皮質におけるタンパク質の発現量を調べた実験データを、PCAにて2次元に次元圧縮する
import urllib.request
import zipfile

# URLを指定
url = "https://storage.googleapis.com/tutor-contents-dataset.aidemy.jp/5030_unsupervised_learning_data.zip"
save_name = url.split('/')[-1]

# ダウンロードする
mem = urllib.request.urlopen(url).read()

# ファイルへ保存
with open(save_name, mode='wb') as f:
    f.write(mem)

# zipファイルをカレントディレクトリに展開する
zfile = zipfile.ZipFile(save_name)
zfile.extractall('.')

# 必要なライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PCAクラスを定義
class PCA:
    def __init__(self):
           pass

    def fit(self, X):
        # 引数Xをインスタンス変数Xに代入
        self.X = X
        # 受け取ったデータXを標準化
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # 標準化したデータの相関行列を計算
        R = np.corrcoef(X.T)
        # 相関行列を固有値分解し、固有値と固有ベクトルを求める
        eigvals, eigvecs = np.linalg.eigh(R)
        # 2次元に圧縮する特徴変換行列を作成
        W = np.c_[eigvecs[:, -1], eigvecs[:, -2]]
        # データXを特徴変換して得たデータをインスタンス変数dataに代入
        self.data = X.dot(W)


# 実験データを読み込み、DataFrameを作成
df = pd.read_csv("./5030_unsupervised_learning_data/Data_Cortex_Nuclear.csv")

# 今回は使用しないタンパク質のデータである、21列目〜80列目を削除
df = df.drop(df.columns[range(21, 81)], axis=1)

# 今回使用するクラスのマウスは計29匹なので、29×15=435個のデータを用いる
# 435行目以降は使わないので434行目までを抽出
df = df.iloc[:435,]

# 欠損値nanを含む行をリストワイズ削除
df = df.dropna(how="any", axis=0)

# 最終列のclassを抽出し、ラベルyとして定義
y = df.loc[:, "class"]

# 0列目のMouseIDと目的変数であるclassを削除し、特徴量Xとして定義
X = df.drop(["MouseID", "class"], axis=1)

# PCAクラスを用いてデータを分析
clf = PCA()
clf.fit(X)


# 圧縮したデータを取得し、matplotlibで表示
X_pca = clf.data
colors = ["r", "g", "b"]

for label, color in zip(y.unique(), colors):
    # 横軸(第一引数)にはX_pcaの0列目を、縦軸(第二引数)にはX_pcaの1列目を表示
    plt.scatter(X_pca.loc[y == label, 0], X_pca.loc[y == label, 1],c=color, label=label)
plt.legend()
plt.show()