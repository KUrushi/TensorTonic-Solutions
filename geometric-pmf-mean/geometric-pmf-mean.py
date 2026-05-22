import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Write code here
    K = np.array(k)
    
    return np.power(1-p, K-1) * p, 1/p
        