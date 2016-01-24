import numpy as np
import matplotlib.pyplot as plt
from k_NN import k_NN

with np.load("TINY_MNIST.npz") as data:
    x, t = data["x"], data["t"]
    x_eval, t_eval = data["x_eval"], data["t_eval"]
        
xed = np.shape(x_eval)
nl = xed[0]
nc = xed[1]

N_array = [ 5, 50, 100, 200, 400, 800]
N_array_d = np.shape(N_array)
n = N_array_d[0]
Err_Nr = np.zeros(n)

l = 0
for nepochs in N_array:
    t_eval_p = np.zeros(nl)
    for i in range(0, nl):
        t_eval_p[i] = k_NN(x, t, x_eval[i],1, nepochs)
    for i in range(0, nl):
        if t_eval_p[i] != t_eval[i]:
            Err_Nr[l] = Err_Nr[l] + 1
    l = l+1
    
plt.plot(N_array, Err_Nr)
plt.show()

print("T size\t\tErr_Nr")
for i in range (0,n):
    print_text = '%d %s %d' % (N_array[i], "\t\t", Err_Nr[i])
    print(print_text)
