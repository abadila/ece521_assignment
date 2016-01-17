# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent


train_x = np.linspace (1.0 , 10.0 , num =100) [:, np.newaxis]
train_y = np.sin( train_x ) + np.power ( train_x , 2) * 0.1 + np.random .randn (100 , 1) * 0.5
yd = np.shape(train_y)
T = yd[0]
eta = 0.001
plt.plot(train_x, train_y)
w = gradient_descent(train_x, train_y, eta)
train_y_p = np.zeros(T, dtype=float)
for i in range(0, T):
    train_y_p[i] = w[0] + w[1] * train_x[i]
plt.plot(train_x, train_y_p)
plt.show()
