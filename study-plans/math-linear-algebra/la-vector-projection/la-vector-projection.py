import numpy as np

def vector_projection(u, v):
    """
    Returns: float64 array, the projection of u onto v.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    return np.dot(u,v) / np.dot(v,v) * v