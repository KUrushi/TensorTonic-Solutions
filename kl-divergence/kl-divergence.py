import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * (np.log(p) - np.log(q+eps)))