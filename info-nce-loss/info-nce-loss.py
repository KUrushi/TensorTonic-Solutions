import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    S = np.dot(Z1, Z2.T) / temperature
    S = S - np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S)
    sum_row = np.sum(exp_S, axis=1)
    return -np.mean(np.log(np.diag(exp_S)/sum_row))