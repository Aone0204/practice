#店舗情報が記述された変数名store_df、商品の売り上げ情報がまとめられている変数名dfの2つのDataFrameが存在します。

#store_dfにはstore (店舗名)とID (店舗に割り当てられているID)がまとめられています。

#dfにはID (store_dfのIDと対応)、product (商品)、price(価格)、quantity(個数)がまとめられています。

#2つのDataFrameを用いて分析します。

import pandas as pd


store_data = {
    "store": ["shibuya", "shinjuku", "yokohama", "meguro", "ikebukuro"],
    "ID": [1, 2, 3, 4, 5]
}
store_df = pd.DataFrame(store_data)  # store_dfを作成

data = {"ID": [1, 2, 3, 3, 2, 1],
        "product": ["banana", "orange", "orange", "grape", "banana", "peach"],
        "price": [200, 1000, 800, 100, 250, 900],
        "quantity": [1, 2, 1, 2, 3, 2]}
df = pd.DataFrame(data)  # dfを作成

print(df)  # dfを出力
print()
print(store_df)  # store_dfを出力
print()


# dfのインデックスが０から４までの要素、カラム名を出力
df_1 = df.iloc[0:5,]
print(df_1)
print()


# df とstore_dfをkeyをIDとして完全外部結合する
df_2 = pd.merge(store_df, df, on = "ID", how="outer")
print(df_2)
print()


# df とstore_dfをkeyをIDとして内部結合する
df_3 = pd.merge(store_df, df, on = "ID", how="inner")
print(df_3)
print()


# df_3とgroupbyメソッドを用いてstore毎のID、price、quantityの平均値を出力
print(df_3.dtypes)
print()
df_4 = df_3.groupby("store").sum()
print(df_4)
print()
df_3 = df_3.drop("product", axis=1)
df_4 = df_3.groupby("store").mean()
print(df_4)
print()


# df_3とdescribeメソッドを用いてID、price、quantityの要約統計量を出力
df_5 = df_3.describe()
print(df_5)