import numpy as np

def hadamard_product(A, B):
    """
    Returns: ndarray, the element-wise product A * B.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    return A * B