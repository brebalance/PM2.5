# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from preprocess import mat_list_train, label_list_train, mat_list_val, label_list_val

rate = 1
para_list = [0.01] * 27
para_list = np.array(para_list).astype(float)
b = 0.01
_y_list = []
error_list = []
grad = []


def cal_y():
    _y_list.clear()
    for i in range(len(mat_list_train)):
        mat_list_train[i] = mat_list_train[i].reshape((27, 1))
        _y = np.dot(para_list, mat_list_train[i]) + b
        _y_list.append(_y)

    # print(np.shape(mat_list_train))
    # print(np.shape(_y_list))


def cal_error():
    error_list.clear()
    for i in range(len(_y_list)):
        error = (label_list_train[i] - _y_list[i][0])
        error_list.append(error)


def cal_grad():
    grad.clear()
    # b的梯度
    g = 0
    for i in range(len(error_list)):
        g = g - 2 * error_list[i]
    g = g / len(error_list)
    grad.append(g)
    # 其他参数的梯度
    for i in range(len(para_list)):
        g = 0
        for j in range(len(error_list)):
            g = g - 2 * error_list[j] * mat_list_train[j][i][0]

        g = g / len(error_list)
        grad.append(g)


m_grad = [0] * 28
m_grad = np.array(m_grad).astype(float)


def cal_m_grad():
    for i in range(len(grad)):
        m_grad[i] = m_grad[i] + grad[i] ** 2


while (True):
    cal_y()
    cal_error()
    cal_grad()
    cal_m_grad()
    error = 0
    for j in range(len(error_list)):
        error = error + (error_list[j] ** 2)
    error = error / len(error_list)

    for j in range(len(grad)):
        if j == 0:
            b = b - rate * grad[j] / (m_grad[j] ** 0.5)
        else:
            para_list[j - 1] = para_list[j - 1] - rate * grad[j] / (m_grad[j] ** 0.5)

    if error < 40:
        break

_y_list.clear()
for i in range(len(mat_list_val)):
    mat_list_val[i] = mat_list_val[i].reshape((27, 1))
    _y = np.dot(para_list, mat_list_val[i]) + b
    _y_list.append(_y)
error_list.clear()
for i in range(len(_y_list)):
    error = (label_list_val[i] - _y_list[i][0])
    error_list.append(error)
error = 0
for j in range(len(error_list)):
    error = error + (error_list[j] ** 2)
error = error / len(error_list)

'''
plt.plot(label_list_val)
plt.plot(_y_list)
plt.show()
'''
