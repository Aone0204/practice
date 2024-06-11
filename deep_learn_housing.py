#fetch_california_housingデータセットを使って住宅の価格予想を行う

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# 出力結果の固定
tf.random.set_seed(0)

%matplotlib inline

# sample_dataからデータセットを読み込む
train = pd.read_csv('sample_data/california_housing_train.csv')  # sample_dataからデータを読み込む

Y = train.median_house_value # yにはMedHouseValが格納
X = train.drop(columns='median_house_value') # xにはMedHouseVal以外の列が格納

# 説明変数のデータから緯度・経度（Latitude・Longitude）のデータを削除
X=X.drop(columns=['latitude','longitude'])

# テストデータとトレーニングデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


model = Sequential()
model.add(Dense(32, input_dim=6))
model.add(Activation('relu'))
#ユニット数128の中間層をmodel.addで追加
model.add(Dense(128))
#活性化関数reluをmodel.addで追加
model.add(Activation('relu'))
model.add(Dense(1))

# 損失関数にmse、最適化関数にadamを採用
model.compile(loss='mse', optimizer='adam')


# モデルを学習
history = model.fit(X_train, y_train,
                    epochs=60,   # エポック数
                    batch_size=1,  # バッチサイズ
                    verbose=1,
                    validation_data=(X_test, y_test) )

# 予測値を出力
y_pred =  model.predict(X_test)# model.predictにX_testのデータを入れて予測値を出力

# 二乗誤差を出力
mse= mean_squared_error(y_test, y_pred)
print("REG RMSE : %.2f" % (mse** 0.5))

# epoch毎の予測値の正解データとの誤差を表す
# バリデーションデータのみ誤差が大きい場合、過学習を起こしている

train_loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=len(train_loss)

plt.plot(range(epochs), train_loss, marker = '.', label = 'train_loss')
plt.plot(range(epochs), val_loss, marker = '.', label = 'val_loss')
plt.legend(loc = 'best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()