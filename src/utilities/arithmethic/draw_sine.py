import numpy as np
import matplotlib.pyplot as plt

n = 1000

freq = 10 # 頻率
stop = .25 # 取樣範圍 0 ~ stop
sample_rate = 2000 # 每秒鐘對樣本的取樣次數

x = np.linspace(0, stop, int(stop * sample_rate), endpoint=False)
y = np.sin(freq * 2 * np.pi * x)
for i in range(1, n, 2):
    y += 4 * np.sin(np.pi * x * i) / ( np.pi * i)

plt.plot(x, y)
plt.show()