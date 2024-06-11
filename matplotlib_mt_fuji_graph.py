#1~3のグラフを合わせると4のグラフが富士山になる

import math
import matplotlib.pyplot as plt
import numpy as np

# 描画用の点を用意しています
x1 = np.linspace(-1, 1, 200)
y1 = [x**4 - x**2 + 6 for x in x1]
x2 = np.linspace(-8, -1, 700)
y2 = [12 / (abs(x) + 1) for x in x2]
x3 = np.linspace(1, 8, 700)
y3 = [12 / (abs(x) + 1) for x in x3]
x4 = np.linspace(-2, 2, 400)
y4 = [1/2 * math.cos(6*x) + 7/2 for x in x4]

# figureオブジェクトを作成
fig = plt.figure()

# axesオブジェクトを2行2列に分割し、左上に (x1,y1) を描画
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x1, y1)
ax1.set_xlim(-8, 8)
ax1.set_ylim(0, 8)


# axesオブジェクトを2行2列に分割し、右上に (x2,y2), (x3,y3) を重ねて描画
# グラフの色は全て赤色
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x2, y2, color="r")
ax2.plot(x3, y3, color="r")
ax2.set_xlim(-8, 8)
ax2.set_ylim(0, 8)


# axesオブジェクトを2行2列に分割し、左下に (x4,y4) を描画
# グラフのスタイルは破線
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x4, y4, linestyle="--")
ax3.set_xlim(-8, 8)
ax3.set_ylim(0, 8)


# axesオブジェクトを2行2列に分割し、右下に (x1,y1), (x2,y2), (x3,y3), (x4,y4) を重ねて描画
# グラフの色は全て青色
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x1, y1, color="b")
ax4.plot(x2, y2, color="b")
ax4.plot(x3, y3, color="b")
ax4.plot(x4, y4, color="b")
ax4.set_xlim(-8, 8)
ax4.set_ylim(0, 8)


plt.show()