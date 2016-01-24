import numpy as np
import matplotlib.pyplot as plt
from reg_gradient_descent import reg_gradient_descent

with np.load ("TINY_MNIST.npz") as data :
    x, t = data ["x"], data["t"]
    x_eval , t_eval = data ["x_eval"], data ["t_eval"] 
L = [100, 200, 400, 800]
shape_L = np.shape(L)
K = shape_L[0]
eta = 0.3 
error = np.zeros((K, 2))
NUM_IT = 100
for i in range(0, K):
    train_x = x[0:L[i], :]
    train_t = t[0:L[i]]
    train_x_shape = np.shape(train_x)
    x_eval_shape = np.shape(x_eval)
    T = x_eval_shape[0]
    N = train_x_shape[1]
    #M = train_x_shape[0]
    w = reg_gradient_descent(train_x, train_t, eta, 0, NUM_IT)
    error[i, 0] = L[i]   
    for j in range(0, T):   
        t_predicted = np.dot(w[1:N+1], np.transpose(x_eval[j, :])) + w[0]  
        if t_predicted <= 0.5:
            t_predicted = 0
        else:
            t_predicted = 1   
        if t_predicted != t_eval[j]:
            error[i, 1] = error[i, 1] + 1              
for i in range(0, K):
    print("%f\t%d" % (error[i, 0], error[i, 1]))
    
