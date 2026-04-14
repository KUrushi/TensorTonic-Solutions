import numpy as np

def matrix_transpose(A):
    """
    Returns: ndarray, the transpose of A.
    """
    A = np.asarray(A)
    output = np.zeros((A.shape[::-1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            output[j,i] = A[i,j]
    return output
    