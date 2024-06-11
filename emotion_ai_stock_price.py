#ITライフハックの記事を感情分析して、株価の予測モデルを構築する
#予測モデルにはk近傍法を用いる。kの値が1~10におけるそれぞれの正答率をプロットする
import urllib.request
import zipfile

# URLを指定
url = "https://storage.googleapis.com/tutor-contents-dataset.aidemy.jp/6050_stock_price_prediction_data.zip"
save_name = url.split('/')[-1]

# ダウンロードする
mem = urllib.request.urlopen(url).read()

# ファイルへ保存
with open(save_name, mode='wb') as f:
    f.write(mem)

# zipファイルをカレントディレクトリに展開する
zfile = zipfile.ZipFile(save_name)
zfile.extractall('.')

# URLを指定
url = "https://storage.googleapis.com/tutor-contents-dataset.aidemy.jp/5050_nlp_data.zip"
save_name = url.split('/')[-1]

# ダウンロードする
mem = urllib.request.urlopen(url).read()

# ファイルへ保存
with open(save_name, mode='wb') as f:
    f.write(mem)

# zipファイルをカレントディレクトリに展開する
zfile = zipfile.ZipFile(save_name)
zfile.extractall('.')

# URLを指定
url = "https://storage.googleapis.com/tutor-contents-dataset.aidemy.jp/6020_negative_positive_data.zip"
save_name = url.split('/')[-1]

# ダウンロードする
mem = urllib.request.urlopen(url).read()

# ファイルへ保存
with open(save_name, mode='wb') as f:
    f.write(mem)

# zipファイルをカレントディレクトリに展開する
zfile = zipfile.ZipFile(save_name)
zfile.extractall('.')



!apt install aptitude
!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
!pip install mecab-python3==0.7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MeCab
import re
import glob
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
記事データの形態素解析を行い、記事ごとにPN値を算出
'''
# MeCabインスタンスの作成、引数を無指定にするとIPA辞書になる
m = MeCab.Tagger('')

# テキストを形態素解析し辞書のリストを返す関数
def get_diclist(text):
    parsed = m.parse(text)      # 形態素解析結果（改行を含む文字列として得られる）
    lines = parsed.split('\n')  # 解析結果を1行（1語）ごとに分割してリスト化
    lines = lines[0:-2]         # 後ろ2行は不要なので削除
    diclist = []
    for word in lines:
        l = re.split('\t|,',word)  # 各行はタブとカンマで区切られているため
        d = {'BaseForm':l[7]}
        diclist.append(d)
    return(diclist)

# 形態素解析結果の単語ごとのdictデータにPN値を追加する関数
def add_pnvalue(diclist_old, pn_dict):
    diclist_new = []
    for word in diclist_old:
        base = word['BaseForm']        # 個々の辞書から基本形を取得
        if base in pn_dict:
            pn = float(pn_dict[base])
        else:
            pn = 'notfound'            # その語がPN Tableになかった場合
        word['PN'] = pn
        diclist_new.append(word)
    return(diclist_new)

# 各記事のPN平均値を求める
def get_mean(dictlist):
    pn_list = []
    for word in dictlist:
        pn = word['PN']
        if pn!='notfound':
            pn_list.append(pn)
    if len(pn_list)>0:
        pnmean = np.mean(pn_list)
    else:
        pnmean=0
    return pnmean

# 取得した記事の読み込み
def load_it():
    created_at = []
    texts  = []
    files = glob.glob("5050_nlp_data/it-life-hack/it-life-hack-*.txt")
    for file in files[:400]: # 実行時間の都合上、読み込むファイルを制限
        with open(file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            time_complex = lines[1].split("T")
            time = dt.strptime(time_complex[0], '%Y-%m-%d')
            text = "".join(lines[2:])
        texts.append(text)
        created_at.append(time)
    return created_at, texts

created_at, texts = load_it()

# 辞書型配列を作成して出力
dicta = {'日付け': created_at, 'texts': texts}
df_corpus = pd.DataFrame(dicta)
# 日付け列を時系列順
df_corpus = df_corpus.sort_values(['日付け'])
# 日付け列中の重複したデータを削除
df_corpus = df_corpus.drop_duplicates(['日付け'])
# 日付け列をインデックスに設定
df_corpus = df_corpus.set_index(['日付け'])

# 極性辞書を読み込み、単語と極性情報を格納
pn_df = pd.read_csv('./6020_negative_positive_data/data/pn_ja.dic',\
                    sep=':',
                    encoding='utf-8',
                    names=('Word','Reading','POS', 'PN')
                   )
word_list = list(pn_df['Word'])
pn_list   = list(pn_df['PN'])
pn_dict = dict(zip(word_list, pn_list))

# 空のリストを作り、記事ごとの平均値を求める
means_list = []
for text in df_corpus['texts']:
    dl_old = get_diclist(text)
    dl_new = add_pnvalue(dl_old, pn_dict)
    pnmean = get_mean(dl_new)
    means_list.append(pnmean)
df_corpus['pn'] = means_list

'''
株価データの終値と、日付けごとの記事データのPN値をテーブルに結合
'''
# chart.csvから株価情報の読み込み
df_chart = pd.read_csv("6050_stock_price_prediction_data/chart.csv")

# indexを日付けにした後、時系列に変換
df_chart["日付け"] = pd.to_datetime(df_chart["日付け"], format='%Y年%m月%d日')
df_chart["終値"] = df_chart["終値"].replace(',', '', regex=True)
# カラムから'始値', '高値', '安値'を取り除いて、日付が古い順に並べる
df_chart = df_chart.drop(['始値', '高値', '安値', '出来高', '前日比%'], axis=1)
# 日付順にソート
df_chart = df_chart.sort_values(['日付け'])

# df_corpusとdf_chartを、日付けをキーにして内部結合
df_table = df_corpus.join(df_chart, how='right')
df_table = pd.merge(df_corpus, df_chart, how="inner", on = "日付け")
# 日付順にソート
df_table = df_table.sort_values(['日付け'])

'''
訓練データ・テストデータに分割
'''
# XにPN値を、yに終値を格納
X = df_table.values[:, 2]
y = df_table.values[:, 3]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
# 訓練データとテストデータの標準化
X_train_std = (X_train - X_train.mean()) / X_train.std()
X_test_std = (X_test - X_train.mean()) / X_train.std()

# df_trainというテーブルを作りそこにindexを日付け、カラム名をpn値、終値にして出力
df_train = pd.DataFrame(
    {'pn': X_train_std,
     '終値': y_train},
    columns=['pn', '終値'],
    index=df_table['日付け'][:len(X_train_std)])

# テストデータについても同様にdf_testというテーブルを作成
df_test = pd.DataFrame(
    {'pn': X_test_std,
     '終値': y_test},
    columns=['pn', '終値'],
    index=df_table['日付け'][len(X_train_std):])

'''
PN値と株価の変化を格納
'''
# 日付を格納
exchange_dates = []

# 1日ごとのpn値の差分を格納する準備
pn_rates = []
pn_rates_diff = []

# 1日ごとの株価の差分を格納する準備
exchange_rates = []
exchange_rates_diff = []

prev_pn = df_train['pn'][0] # typeはfloat
prev_exch = float(df_train['終値'][0])

# 訓練データの数だけPN値・株価の変化を算出
for i in range(len(X_train_std)):
    time = df_train.index[i]   # 日付け
    pn_val = df_train['pn'][i]   # 訓練データのPN値
    exch_val = float(df_train['終値'][i])  # 訓練データの終値

    exchange_dates.append(time)  # 日付
    pn_rates_diff.append(pn_val - prev_pn)   # PN値の変化
    exchange_rates_diff.append(exch_val - prev_exch)   # 株価の変化

    # 前日のPN値、終値を更新
    prev_pn = pn_val
    prev_exch = exch_val

'''
3日間ごとのPN値と株価の変化を表示
'''
INPUT_LEN = 3
data_len = len(pn_rates_diff)

# 説明変数を格納する準備
tr_input_mat = []
# 目的変数を格納する準備
tr_angle_mat = []

# 直近3日間なので、INPUT_LENから開始
for i in range(INPUT_LEN, data_len):
    tmp_arr = []
    # i日目の直近3日間の株価とネガポジの変化を格納
    for j in range(INPUT_LEN):
        tmp_arr.append(exchange_rates_diff[i-INPUT_LEN+j])
        tmp_arr.append(pn_rates_diff[i-INPUT_LEN+j])
    tr_input_mat.append(tmp_arr)

    # i日目の株価の上下（プラスなら1、マイナスなら0）を格納
    if exchange_rates_diff[i] >= 0:
        tr_angle_mat.append(1)
    else:
        tr_angle_mat.append(0)

# numpy配列に変換して結果を代入
train_feature_arr = np.array(tr_input_mat)
train_label_arr = np.array(tr_angle_mat)

'''
訓練データと同様に、テストデータに関する予測準備
'''
# 日付を格納
exchange_dates_test = []

# 1日ごとのpn値の差分を格納する準備
pn_rates_test = []
pn_rates_diff_test = []

# 1日ごとの株価の差分を格納する準備
exchange_rates_test = []
exchange_rates_diff_test = []

# 訓練データの数だけPN値・株価の変化を算出
for i in range(len(X_test_std)):
    time = df_test.index[i]   # 日付け
    pn_val = df_test['pn'][i]   # 訓練データのPN値
    exch_val = float(df_test['終値'][i])  # 訓練データの終値

    exchange_dates_test.append(time)  # 日付
    pn_rates_diff_test.append(pn_val - prev_pn)   # PN値の変化
    exchange_rates_diff_test.append(exch_val - prev_exch)   # 株価の変化

    # 前日のPN値、終値を更新
    prev_pn = pn_val
    prev_exch = exch_val


data_len = len(pn_rates_diff_test)
test_input_mat = []
test_angle_mat = []

for i in range(INPUT_LEN, data_len):
    test_arr = []
    for j in range(INPUT_LEN):
        test_arr.append(exchange_rates_diff_test[i - INPUT_LEN + j])
        test_arr.append(pn_rates_diff_test[i - INPUT_LEN + j])
    test_input_mat.append(test_arr)  # i日目の直近3日間の株価とネガポジの変化

    if exchange_rates_diff[i] >= 0:  # i日目の株価の上下（プラスなら1、マイナスなら0）
        test_angle_mat.append(1)
    else:
        test_angle_mat.append(0)

test_feature_arr = np.array(test_input_mat)
test_label_arr = np.array(test_angle_mat)


k_list = [i for i in range(1, 11)]

accuracy = []

for k in k_list :
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(train_feature_arr, train_label_arr)
  accuracy.append(model.score(test_feature_arr, test_label_arr))

plt.plot(k_list, accuracy)
plt.xlabel("n_neighbor")
plt.ylabel("accuracy")
plt.title("accuracy by changing n_neighbor")
plt.show()
