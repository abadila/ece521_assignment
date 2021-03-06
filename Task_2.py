import numpy as np
import matplotlib.pyplot as plt
from k_NN import k_NN

with np.load("TINY_MNIST.npz") as data:
    x, t = data["x"], data["t"]
    x_eval, t_eval = data["x_eval"], data["t_eval"]
        
xed = np.shape(x_eval)
nl = xed[0]
nc = xed[1]

N = 800
k_array = [1, 3, 5, 7, 21, 101, 401]
k_array_d = np.shape(k_array)
n = k_array_d[0]
Err_Nr = np.zeros(n)

l = 0
for k in k_array:
    t_eval_p = np.zeros(nl)
    for i in range(0, nl):
        t_eval_p[i] = k_NN(x, t, x_eval[i], k, N)
    for i in range(0, nl):
        if t_eval_p[i] != t_eval[i]:
            Err_Nr[l] = Err_Nr[l] + 1
    l = l+1
    
plt.plot(k_array, Err_Nr)
plt.show()

print("k\t\tErr_Nr")
for i in range (0,n):
    print_text = '%d %s %d' % (k_array[i], "\t\t", Err_Nr[i])
    print(print_text)
