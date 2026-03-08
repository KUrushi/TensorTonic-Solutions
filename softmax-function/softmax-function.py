import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    x_max = np.max(x, axis=-1, keepdims=True)
    x = x - x_max
    return np.exp(x) / np.sum(np.exp(x),axis=-1,keepdims=True)