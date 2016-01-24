import numpy as np

def reg_gradient_descent(x, y, eta, lamda, NUM_IT):
    xd = np.shape(x)
    yd = np.shape(y)
    N = xd[1]
    T = yd[0]
    w = np.random.rand(N + 1)
    y_hat = np.zeros(T, dtype=float)
    for l in range(0, NUM_IT):
        for k in range(0, T):
            y_hat[k] = w[0]
            for i in range(1, N + 1):
                y_hat[k] = y_hat[k] + w[i] * x[k, i-1]
        delta = np.zeros(N+1, dtype=float)
        for k in range(0, T):            
            delta[0] = delta[0] + (y[k,0] - y_hat[k])/T 
        delta[0] = delta[0] - lamda * w[0]             
        for i in range(1, N+1):
            for k in range(0, T):           
                delta[i] = delta[i] +  (y[k,0] - y_hat[k])*x[k, i-1]/T
            delta[i] = delta[i] - lamda * w[i]              
        for i in range(0, N + 1):
            w[i] = w[i] + eta * delta[i]
    return w    
