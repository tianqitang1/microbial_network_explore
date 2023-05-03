import numpy as np

def random_baseline(arr):
    n = arr.shape[1]
    return np.random.rand(n, n)