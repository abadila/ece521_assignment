# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from reg_gradient_descent import reg_gradient_descent

NUM_IT = 1000

train_x_1 = np.linspace (1.0 , 10.0 , num =100) [:, np. newaxis]
train_x_2 = [x**2 for x in train_x_1]
train_x_3 = [x**3 for x in train_x_1]
train_x_4 = [x**4 for x in train_x_1]
train_x_5 = [x**5 for x in train_x_1]

train_mean = np.zeros(5)
train_mean[0] = np.mean(train_x_1)
train_mean[1] = np.mean(train_x_2)
train_mean[2] = np.mean(train_x_3)
train_mean[3] = np.mean(train_x_4)
train_mean[4] = np.mean(train_x_5)

train_std = np.zeros(5)
train_std[0] = np.std(train_x_1)
train_std[1] = np.std(train_x_2)
train_std[2] = np.std(train_x_3)
train_std[3] = np.std(train_x_4)
train_std[4] = np.std(train_x_5)

train_x_1_n = [(x-train_mean[0])/train_std[0] for x in train_x_1]
train_x_2_n = [(x-train_mean[1])/train_std[1] for x in train_x_2]
train_x_3_n = [(x-train_mean[2])/train_std[2] for x in train_x_3]
train_x_4_n = [(x-train_mean[3])/train_std[3] for x in train_x_4]
train_x_5_n = [(x-train_mean[4])/train_std[4] for x in train_x_5]

train_x = np.zeros((100,5))

train_x[:, 0] = train_x_1_n
train_x[:, 1] = train_x_2_n
train_x[:, 2] = train_x_3_n
train_x[:, 3] = train_x_4_n
train_x[:, 4] = train_x_5_n

train_y = np.sin( train_x_1 ) + np.power ( train_x_1 , 2) * 0.1 + np.random .randn (100 , 1) * 0.5
eta = 0.01
plt.plot(train_x_1, train_y)
w = reg_gradient_descent(train_x, train_y, eta, 0, NUM_IT)
train_y_p = np.zeros(100)
for i in range(0, 100):
    train_y_p[i] = w[0]
    for j in range(1, 6):
        train_y_p[i] = train_y_p[i] + w[j] * ((train_x_1[i] ** j - train_mean[j-1])/train_std[j-1])

plt.plot(train_x_1, train_y_p)
plt.show()
