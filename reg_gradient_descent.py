import numpy as np

def reg_gradient_descent(x, y, eta, lamda):
    xd = np.shape(x)
    yd = np.shape(y)
    N = xd[1]
    T = yd[0]
    w = np.random.rand(N + 1)
    delta = np.zeros(N + 1, dtype=float)
    y_hat = np.zeros(T, dtype=float)
    NUM_IT = 10000
    for l in range(0, NUM_IT):
        for k in range(0, T):
            y_hat[k] = w[0]
            for i in range(1, N + 1):
				print i
                y_hat[k] = y_hat[k] + w[i] * x[k, i-1]
        for k in range(0, T):            
            delta[0] = delta[0] + (2 * (y[k,0] - y_hat[k]))/T - lamda * w[0]          
        for i in range(1, N + 1):
            for k in range(0, T):           
                delta[i] = delta[i] + (2 * (y[k,0] - y_hat[k]) * x[k, i-1])/T - lamda * w[i]          
        for i in range(0, N + 1):
            w[i] = w[i] + eta * delta[i]
    return w            
