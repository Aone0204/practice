#ユニクロの株価データからStock Tradingの推移を抽出し、Matplotlibを利用してグラフを出力
#日付順にソートし直した株価データをエクセルファイルに書き込んで保存する

import urllib.request
import zipfile

# URLを指定
url = "https://storage.googleapis.com/tutor-contents-dataset.aidemy.jp/5130_rnn_lstm_data.zip"
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
import matplotlib.pyplot as plt

# 株価データを読み込む
stock_data = pd.read_csv('./5130_rnn_lstm_data/uniqlo_training_stocks_2012-2017.csv')


#　stock_dataの値を出力 Dateが日付順になっていないことを確認
print(stock_data.head(10))

# Dateをstr型からdatetime型に変換
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
# Dateを日付順にソート
sorted_stock_data = stock_data.sort_values(['Date'])


# Stock Trading列を抽出
stock_trading = sorted_stock_data['Stock Trading']

# plot()メソッドでプロット
stock_trading.plot()

# グラフを出力
plt.show()

# sorted_stock_dataを'sorted_stock_data'という名前でExcelファイルへ書き出す
sorted_stock_data.to_excel('sorted_stock_data.xlsx')
print(sorted_stock_data.head(10))