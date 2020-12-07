# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from train_model import para_list, b

data = pd.read_csv("test.csv")

del data['id_0']
del data['AMB_TEMP']
data = data.replace(['NR', [-1.0]])
data = np.array(data).astype(float)

test = []
for i in range(-1, 4319, 18):
    temp = (data[i + 4]).copy()
    temp = np.hstack((temp, data[i + 9]))
    temp = np.hstack((temp, data[i + 12]))
    temp = temp.reshape((27, 1))
    test.append(temp)

_y_list = []
for i in range(len(test)):
    _y = np.dot(para_list, test[i]) + b
    _y_list.append(_y)

y_arr = pd.read_csv("ans.csv")
del y_arr['id']
y_arr = np.array(y_arr).astype(float)

y_list = y_arr.reshape(240, 1)
plt.plot(y_list)
plt.plot(_y_list)
plt.show()
