#車の売上データ「sales_car」をSARIMAモデルを用いて予測
import urllib.request
import zipfile

# URLを指定
url = "https://storage.googleapis.com/tutor-contents-dataset.aidemy.jp/5060_tsa_data.zip"
save_name = url.split('/')[-1]

# ダウンロードする
mem = urllib.request.urlopen(url).read()

# ファイルへ保存
with open(save_name, mode='wb') as f:
    f.write(mem)

# zipファイルをカレントディレクトリに展開
zfile = zipfile.ZipFile(save_name)
zfile.extractall('.')


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

# 1.データの読み込み
sales_car = pd.read_csv("./5060_tsa_data/monthly-car-sales-in-quebec-1960.csv")

# 2.データの整理
# indexに期間("1960-01-31"から"1968-12-31")を代入
index = pd.date_range("1960-01-31", "1968-12-31", freq = "M")

# sales_carのインデックスにindexを代入
sales_car.index = index

# sales_carの"Month"カラムを削除
del sales_car["Month"]


# orderの最適化関数
def selectparameter(DATA, s):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]
    parameters = []
    BICs = np.array([])
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(DATA,
                                                order=param,
                                                seasonal_order=param_seasonal)
                results = mod.fit()
                parameters.append([param, param_seasonal, results.bic])
                BICs = np.append(BICs, results.bic)
            except:
                continue
    return parameters[np.argmin(BICs)]

#print(selectparameter(sales_car, 12))

# 5.モデルの構築

# SARIMAモデルを用いて時系列解析をする
# 周期は月ごとのデータであることも考慮してs=12
# orderはselectparameter関数の0インデックス, seasonal_orderは1インデックスに格納
best_params = selectparameter(sales_car, 12)
SARIMA_sales_car = sm.tsa.statespace.SARIMAX(sales_car, order = selectparameter(sales_car, 12)[0], seasonal_order = selectparameter(sales_car, 12)[1]).fit()



# 予測
# predに予測期間での予測値を代入
pred = SARIMA_sales_car.predict("1968-01-31", "1972-01-31")

# グラフを可視化。予測値は赤色でプロット
plt.plot(sales_car)
plt.plot(pred, color = "r")
plt.show()
