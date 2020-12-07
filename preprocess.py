# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")

del data['dates']
del data['factors']

data = data.replace(['NR', [-1.0]])
arr = np.array(data).astype(float)

# 经查阅相关资料，仅考虑NO，SO2，PM2.5三项变量进行预测

temp = (arr[4]).copy()
for i in range(18, 4320, 18):
    temp = np.hstack((temp, arr[i + 4]))
res = temp.copy()

temp = (arr[9]).copy()
for i in range(18, 4320, 18):
    temp = np.hstack((temp, arr[i + 9]))
res = np.vstack((res, temp))

temp = (arr[12]).copy()
for i in range(18, 4320, 18):
    temp = np.hstack((temp, arr[i + 12]))
res = np.vstack((res, temp))

# 均匀选取四分之一的数据作为validation

mat_list = []
label_list = []
for i in range(len(res[0]) - 9):
    mat = np.vstack((res[0, i:i + 9], res[1, i:i + 9], res[2, i:i + 9]))
    mat_list.append(mat)
    label_list.append(res[1, i + 9])

mat_list_train = []
label_list_train = []
mat_list_val = []
label_list_val = []
for i in range(len(mat_list)):
    if i % 4 == 0:
        mat_list_val.append(mat_list[i])
        label_list_val.append(label_list[i])
    else:
        mat_list_train.append(mat_list[i])
        label_list_train.append(label_list[i])


