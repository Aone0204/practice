#以下の処理を関数 Pud_ding に実装する。

#引数 numbers で与えられたリストの値1～35の全ての値について
#・5 で割り切れる場合、「数値:Pud」と表示する
#・7 で割り切れる場合、「数値:ding」と表示する
#・5 と7 で割り切れる場合、「数値:Pudding」と表示する
#・それ以外の場合は、そのままの数字を表示する

def Pud_ding(numbers):
    # ここに処理を記述してください
  for numbers in range(1, 36, 1):
    if numbers % 35 == 0 :
      print(str(numbers) + ":Pudding")
    elif numbers % 5 == 0 :
      print(str(numbers) + ":Pud")
    elif numbers % 7 == 0 and numbers % 7 == 0 :
      print(str(numbers) + ":ding")
    else:
      print(numbers)


# データの定義
numbers = list(range(1,36))
# 関数の実行
Pud_ding(numbers)