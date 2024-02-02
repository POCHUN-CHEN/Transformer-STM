# 計算提供的數據的平均值和標準差
import numpy as np

data = [370.44, 369.82, 368.83, 365.57, 369.72]
mean = np.mean(data)
std_dev = np.std(data)
print(mean+std_dev)
print(mean-std_dev)