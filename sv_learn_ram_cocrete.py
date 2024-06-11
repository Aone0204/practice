#コンクリートの圧縮強度について回帰分析を行う
#複数のモデル（線形回帰, ridge回帰, lasso回帰, ElasricNet回帰）で回帰分析を行う
#それぞれのモデルで決定係数を算出
#最もよい決定係数を算出したモデルとその値を出力

import urllib.request
import zipfile

# URLを指定
url = "https://storage.googleapis.com/tutor-contents-dataset.aidemy.jp/5010_regression_data.zip"
save_name = url.split('/')[-1]

# ダウンロードする
mem = urllib.request.urlopen(url).read()

# ファイルへ保存
with open(save_name, mode='wb') as f:
    f.write(mem)

# zipファイルをカレントディレクトリに展開する
zfile = zipfile.ZipFile(save_name)
zfile.extractall('.')

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

# データの読み込み
concrete_data = pd.read_excel("./5010_regression_data/Concrete_Data.xls")

concrete_train_X, concrete_test_X, concrete_train_y, concrete_test_y = train_test_split(
    concrete_data.drop('Concrete compressive strength(MPa, megapascals) ', axis=1),
    concrete_data['Concrete compressive strength(MPa, megapascals) '], random_state=42)



model = LinearRegression()
model.fit(concrete_train_X, concrete_train_y)
R2_LR = model.score(concrete_test_X, concrete_test_y)

model = Lasso()
model.fit(concrete_train_X, concrete_train_y)
R2_La = model.score(concrete_test_X, concrete_test_y)

model = Ridge()
model.fit(concrete_train_X, concrete_train_y)
R2_Ri = model.score(concrete_test_X, concrete_test_y)

model = ElasticNet(l1_ratio=1)
model.fit(concrete_train_X, concrete_train_y)
R2_EN = model.score(concrete_test_X, concrete_test_y)

model_series = pd.Series({"線形回帰": R2_LR,
                         "ラッソ回帰": R2_La,
                         "リッジ回帰": R2_Ri,
                         "ElasricNet回帰": R2_EN})
print(model_series)
print("")
print("最もよい決定係数：", str(model_series.idxmax()), str(model_series.max()))