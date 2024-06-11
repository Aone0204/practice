
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