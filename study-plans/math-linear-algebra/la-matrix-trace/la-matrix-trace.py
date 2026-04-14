import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    A = np.asarray(A)
    output = 0
    for i in range(A.shape[0]):
        output += A[i,i]

    return output