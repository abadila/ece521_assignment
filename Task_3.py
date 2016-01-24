# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from reg_gradient_descent import reg_gradient_descent

NUM_IT = 1000
train_x = np.linspace (1.0 , 10.0 , num =100) [:, np.newaxis]
train_y = np.sin( train_x ) + np.power ( train_x , 2) * 0.1 + np.random .randn (100 , 1) * 0.5
eta = 0.01
plt.plot(train_x, train_y)
w = reg_gradient_descent(train_x, train_y, eta, 0, NUM_IT)
train_y_p = np.zeros(100, dtype=float)
for i in range(0, 100):
    train_y_p[i] = w[0] + w[1] * train_x[i]
plt.plot(train_x, train_y_p)
plt.show()

