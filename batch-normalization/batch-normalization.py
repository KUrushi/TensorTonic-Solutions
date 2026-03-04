import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    mu = np.mean(x, axis=1, keepdims=True)

    if x.ndim == 2:
        axis = 0
        gamma = gamma.reshape(1, -1)
        beta = beta.reshape(1,-1)
    elif x.ndim == 4:
        axis = (0, 2, 3)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

    
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.var(x,axis=axis, keepdims=True)
    x_norm = (x-mu) / np.sqrt(sigma+eps)
    y = gamma * x_norm + beta
    
    return y