import numpy as np

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def k_NN(xt, yt, x, k, N):
    xtd = np.shape(xt)
    nc = xtd[1]
    NNx = np.zeros((k, nc))
    NNd = np.repeat(10., k)
    NNy = np.zeros(k)
    
    for i in range(0,k):
        d = dist(xt[i], x)
        for j in range(0,k):
            if d<=NNd[j]:
                for l in range(0,k-j-1):
                    NNx[k-1-l] = NNx[k-2-l]
                    NNd[k-1-l] = NNd[k-2-l]
                    NNy[k-1-l] = NNy[k-2-l]
                NNd[j] = d
                NNx[j] = xt[i]
                NNy[j] = yt[i]
                break
    
    for i in range(k,N):
        d = dist(xt[i], x)
        if d <= NNd[k-1]:
            for j in range(0,k):
                if d<=NNd[j]:
                    for l in range(0,k-j-1):
                        NNx[k-1-l] = NNx[k-2-l]
                        NNd[k-1-l] = NNd[k-2-l]
                        NNy[k-1-l] = NNy[k-2-l]
                    NNd[j] = d
                    NNx[j] = xt[i]
                    NNy[j] = yt[i]
                    break
    
    y = 0.
    for i in range(0, k):
        y = y + NNy[i]/k
    if y<0.5:
        y=0.
    else:
        y=1
    return y
