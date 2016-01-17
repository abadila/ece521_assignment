import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent


train_x = np.linspace (1.0 , 10.0 , num =100) [:, np. newaxis]
train_y = np.sin( train_x ) + np.power ( train_x , 2) * 0.1 + np.random .randn (100 , 1) * 0.1
train_x = [train_x train_x**2 train_x**3 train_x**4 train_x**5]
eta = 0.01
plt.plot(train_x, train_y)
w = gradient_descent(train_x, train_y, eta)
plt.plot(train_x, train_x * w)
plt.show()
