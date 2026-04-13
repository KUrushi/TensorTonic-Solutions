import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    norm = a_norm * b_norm
    if norm == 0.0:
        return 0.0
    return np.dot(a, b.T) / norm