import numpy as np

def gradient_descent(x, y, eta):
    N = x.shape[1]
    print N
    J = y.shape[0]
    w = np.random.random((J, 1))
    delta = np.zeros((J, 1))
    y_hat = np.zeros((J, 1))
    NUM_IT = 10000
    for l in range(0, NUM_IT):
        for k in range(1, N + 1):
            y_hat[k] = w[0]
            for n in range(1, N + 1):
                for i in range(1, N + 1):
                    y_hat[k] = y_hat[k] + w[i] * x[n, i]
                for n in range(1, N + 1):            
                    delta[0] = delta[0] + 2 / N * (y[n] - y_hat[n])             
            for i in range(1, N + 1):
                for n in range(1, N + 1):
                    delta[i] = delta[i] + 2 / N * (y[n] - y_hat[n]) * x[n, i]
            for i in range(0, N + 1):
                w[i] = w[i] + eta * delta[i]
    return w            
