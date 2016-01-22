import numpy as np
from reg_gradient_descent import reg_gradient_descent

with np.load ("TINY_MNIST.npz") as data :
    x, t = data ["x"], data["t"]
    x_eval , t_eval = data ["x_eval"], data ["t_eval"] 
train_x = x[0:50,:]
train_t = t[0:50]
lamda = [0, 0.0001, 0.001, 0.01, 0.1, 0.5]
lamda_shape = np.shape(lamda)
train_x_shape = np.shape(train_x)
T = train_x_shape[0]
L = lamda_shape[0]
eta = 0.01
E = np.zeros((L, 1))
E_final = np.zeros((L, 2))
for i in range(0, L):
    w = reg_gradient_descent(train_x, train_t, eta, lamda[i])
    w_shape = np.shape(w)
    N = w_shape[0]  
    #print np.shape(np.transpose(w))
    #print np.shape(train_x)
    #for j in range(0, T):         
    #E[i, 1] = 1 / (2 * T) * (np.dot(np.transpose(w),train_x[i, :]) + w[0] - train_t[i]) ^ 2 + lamda[i] / 2 * np.dot(np.transpose(w), w)
    #E_final[i, 1] = lamda[i]
    #E_final[i, 2] = E[i]
#print E_final
