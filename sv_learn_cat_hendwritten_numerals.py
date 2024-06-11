#手書きの数字の認識・分類をする
# 分類器は非線形のサポートベクターマシン（SVM）を使用
#非線形SVM（SVC)のハイパーパラメータをグリッドサーチとランダムサーチを利用して決定
#各調整したパラメーターとその値、及び評価結果を出力

import scipy.stats
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#手書き数字の画像データの読み込み
digits_data = load_digits()


#読み込んだ digits_data の内容確認
#print(dir(digits_data))

#読み込んだ digits_data の画像データの確認
#import matplotlib.pyplot as plt
#fig , axes = plt.subplots(2,5,figsize=(10,5),
#                         subplot_kw={'xticks':(),'yticks':()})
#for ax,img in zip(axes.ravel(),digits_data.images):
#    ax.imshow(img)
#plt.show()


# グリッドサーチ：ハイパーパラメーターの値の候補を設定
model_param_set_grid = {SVC(): {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                                "decision_function_shape": ["ovr", "ovo"],
                                "random_state": [42]}}


# ランダムサーチ：ハイパーパラメーターの値の候補を設定
model_param_set_random =  {SVC(): {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                                   "C": scipy.stats.uniform(0.00001, 1000),
                                   "decision_function_shape": ["ovr", "ovo"],
                                   "random_state": [42]}}

#トレーニングデータ、テストデータの分離
train_X, test_X, train_y, test_y = train_test_split(digits_data.data, digits_data.target, random_state=0)

#条件設定
max_score = 0

#グリッドサーチ
for model, param in model_param_set_grid.items():
    clf = GridSearchCV(model, param)
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    score = f1_score(test_y, pred_y, average="micro")

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

print("グリッドサーチ")
print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))
print()

#条件設定
max_score = 0

#ランダムサーチ
for model, param in model_param_set_random.items():
    clf =RandomizedSearchCV(model, param)
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    score = f1_score(test_y, pred_y, average="micro")

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

print("ランダムサーチ")
print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))