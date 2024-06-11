#学習データに出現する単語をテキストの特徴量とし、ランダムフォレストの分類器を学習
#livedoor newsコーパスでの精度を評価する
#テキストの特徴量を名詞、動詞、形容詞、形容動詞のみに限定した時のモデルの精度を評価する
import urllib.request
import zipfile

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

import glob
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from janome.tokenizer import Tokenizer


def load_livedoor_news_corpus():
    category = {
        'dokujo-tsushin': 1,
        'it-life-hack': 2,
        'kaden-channel': 3,
        'livedoor-homme': 4,
        'movie-enter': 5,
        'peachy': 6,
        'smax': 7,
        'sports-watch': 8,
        'topic-news': 9
    }
    docs = []
    labels = []

    for c_name, c_id in category.items():
        files = glob.glob("./5050_nlp_data/{c_name}/{c_name}*.txt".format(c_name=c_name))

        text = ''
        for file in files[:30]: # 実行時間の関係上読み込むデータを制限
            with open(file, 'r', errors='ignore') as f:
                lines = f.read().splitlines()

                # 1,2行目に書いたあるURLと時間は関係ないので取り除く
                url = lines[0]
                datetime = lines[1]
                subject = lines[2]
                body = "".join(lines[3:])
                text = subject + body

            docs.append(text)
            labels.append(c_id)

    return docs, labels


docs, labels = load_livedoor_news_corpus()

# indices は0からドキュメントの数までの整数をランダムに並べ替えた配列
random.seed()
indices = list(range(len(docs)))
# 9割をトレーニングデータとする
separate_num = int(len(docs) * 0.9)

random.shuffle(indices)

train_data = [docs[i] for i in indices[0:separate_num]]
train_labels = [labels[i] for i in indices[0:separate_num]]
test_data = [docs[i] for i in indices[separate_num:]]
test_labels = [labels[i] for i in indices[separate_num:]]

# テキストを分割する関数
t=Tokenizer()
def tokenize1(text):
    tokens = t.tokenize(text)
    noun = []
    for token in tokens:
        noun.append(token.surface)
    return noun

# Tf-idfを用いてtrain_dataをベクトル化し、引数tokenizerにtokenize1を指定
vectorizer = TfidfVectorizer(tokenizer=tokenize1)
train_matrix = vectorizer.fit_transform(train_data)


# ランダムフォレストを用いて分類する
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_matrix, train_labels)

# テストデータを変換
test_matrix = vectorizer.transform(test_data)

# 分類結果を表示
print(clf.score(train_matrix, train_labels))
print(clf.score(test_matrix, test_labels))

# 単語の抽出
def tokenize2(text):
    tokens = t.tokenize(text)
    noun = []
    for token in tokens:
        # 「名詞」「動詞」「形容詞」「形容動詞」を取り出す
        part_of_speech = token.part_of_speech.split(",")[0]
        if part_of_speech == "名詞" or part_of_speech == "動詞" or part_of_speech == "形容詞" or part_of_speech == "形容動詞" :
          noun.append(token.surface)










    return noun


# 単語を抽出して学習
t = Tokenizer()
vectorizer = TfidfVectorizer(tokenizer=tokenize2)
train_matrix = vectorizer.fit_transform(train_data)
test_matrix = vectorizer.transform(test_data)
clf.fit(train_matrix, train_labels)

# 結果を表示
print(clf.score(train_matrix, train_labels))
print(clf.score(test_matrix, test_labels))