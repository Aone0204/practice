#irisのデータセットを学習し分類。正解率を求める。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# (1.データを与える)
# Irisデータセットをロード
iris = datasets.load_iris()
# 3,4列目の特徴量を抽出
X = iris.data[:, [2,3]]
# クラスラベルを取得
y = iris.target

# (2.X_train, X_test, y_train, y_testにデータを格納し、テストデータを30%に指定、データセットが毎回変わらないよう指定)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# (3.機械学習アルゴリズムSVMを使用し、学習用のデータと結果を学習している)
svc = svm.SVC(C=1, kernel='rbf', gamma=0.001)
svc.fit(X_train, y_train)

# (4.テストデータで予測し、正解率を求めて制度評価をしている)
y_pred = svc.predict(X_test)
print ("Accuracy: %.2f"%accuracy_score(y_test, y_pred))