import numpy as np

def matrix_multiply(A, B):
    """
    Returns: 2-D float64 array, the matrix product A @ B.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    return A @ B