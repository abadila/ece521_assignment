import numpy as np
import matplotlib.pyplot as plt
from reg_gradient_descent import reg_gradient_descent

with np.load ("TINY_MNIST.npz") as data :
    x, t = data ["x"], data["t"]
    x_eval , t_eval = data ["x_eval"], data ["t_eval"] 
train_x = x[0:50,:]
train_t = t[0:50]
train_x_shape = np.shape(train_x)
x_eval_shape = np.shape(x_eval)
T = x_eval_shape[0]
N = train_x_shape[1]
M = train_x_shape[0]
eta = 0.1
NUM_IT = [1, 10, 100, 1000, 5000] 
shape_NUM_IT = np.shape(NUM_IT)
L = shape_NUM_IT[0] 
error = np.zeros((L, 3))
for i in range(0, L):
    w = reg_gradient_descent(train_x, train_t, eta, 0, NUM_IT[i])
    error[i, 0] = NUM_IT[i]
    for j in range(0, M):   
        t_predicted_train = np.dot(w[1:N+1], np.transpose(train_x[j, :])) + w[0]  
        if t_predicted_train <= 0.5:
            t_predicted_train = 0
        else:
            t_predicted_train = 1   
        if t_predicted_train != train_t[j]:
            error[i, 1] = error[i, 1] + 1 
    for j in range(0, T):   
        t_predicted_eval = np.dot(w[1:N+1], np.transpose(x_eval[j, :])) + w[0]  
        if t_predicted_eval <= 0.5:
            t_predicted_eval = 0
        else:
            t_predicted_eval = 1   
        if t_predicted_eval != t_eval[j]:
            error[i, 2] = error[i, 2] + 1              
plt.subplot(2, 1, 1)
plt.plot(error[:, 0], error[:, 1])
plt.title('Number of Training Errors vs. Number of Epochs')
plt.subplot(2, 1, 2)
plt.plot(error[:, 0], error[:, 2])
plt.title('Number of Validation Errors vs. Number of Epochs')
plt.show()
