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
N = train_x_shape[1]
eta = 0.01
error = np.zeros((L, 2))
#N = w_shape[0] 
#M = eval_shape[0]  
for i in range(0, L): 
    w = reg_gradient_descent(train_x, train_t, eta, lamda[i])
    error[i, 0] = lamda[i]
    for j in range(1, N):  
        for k in range(0, T): 
            w_shape = np.shape(w)
            M = w_shape[0]    
            error[i, 1] = error[i, 1] + (np.dot(w[1:M], x_eval[j]) + w[0] - t_eval[j]) ** 2 
print error
