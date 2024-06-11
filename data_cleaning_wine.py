#ワインの種類とワインの成分のデータについて
#欠損値を含んだテーブルデータに対して欠損値の処理を行い、機械学習の結果の精度を比較
#以下の欠損値データ処理をコードに組み入れる
#「リストワイズ削除」「欠損値に０を代入」「欠損値に平均値を代入」「欠損値に前の行の値を代入」
#それぞれ学習を実行し、結果を比較

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

#データのロード
wine_data = datasets.load_wine()
data = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)  # 説明変数のDataFrmeを設定
target = pd.DataFrame(wine_data.target, columns=['target'])  # 目的変数のDataFrmeを設定

# 説明変数のDataFraemと目的変数のDataFrameをつなげて　wine_df を作成
wine_df = pd.concat([data,target],axis=1)
wine_df = wine_df.drop(columns=['magnesium','proline']) # 値のスケールが異なる'magnesium','proline'の列を削除
print(wine_df.head())

#欠損値のあるデータの作成
# 以下のデータを欠損
# 10. color_intensity 色の強さ
# 11. hue 色合い
# 12. od280/od315_of_diluted_wines ワインの希釈度合い
np.random.seed(32)

mask5 = np.random.rand(len(wine_df))<0.05 # 5% のデータを欠損させる
mask3 = np.random.rand(len(wine_df))<0.03 # 3% のデータを欠損させる
wine_df.loc[mask5,['color_intensity']] = np.nan # color_intensityのデータを5%欠損させる
wine_df.loc[mask3,['hue']] = np.nan # hueのデータを3%欠損させる
wine_df.loc[mask5,['od280/od315_of_diluted_wines']] = np.nan # od280/od315_of_diluted_winesのデータを5%欠損させる

#欠損値の状況を確認します。
print(wine_df.isnull().sum())

# 欠損値を埋める
#　dropnaを用いてリストワイズ削除
wine_df_listwise = wine_df.dropna()

# 欠損値を埋める
# fillnaを用いてNaNの部分に０を代入
wine_df_zero = wine_df.fillna(0)

# 欠損値を埋める
# fillnaを用いてNaNの部分に、NaNの入っている前(上)の行の値を代入
wine_df_ffill = wine_df.fillna(method = "ffill")

#　欠損値を埋める
# fillnaを用いてNaNの部分にその列の平均値を代入
wine_df_mean =  wine_df.fillna(wine_df.mean())

# 欠損値の処理を行なったデータを使ってRandomForestClassifierで学習を行う
wine_df_all = [wine_df_listwise,wine_df_zero,wine_df_ffill,wine_df_mean]

for i,wine_df_tmp in enumerate(wine_df_all):
    y = wine_df_tmp["target"]
    X = wine_df_tmp.drop("target", axis=1)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 学習
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if i == 0 : df_type = 'listwise'
    if i == 1 : df_type = 'zero'
    if i == 2 : df_type = 'ffill'
    if i == 3 : df_type = 'mean'

    print("{:12s}を用いた欠損値補完:  accuracy_score = {:<.3f}".format(df_type,accuracy_score(y_test, y_pred)))