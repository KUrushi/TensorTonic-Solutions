import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.sum(np.power((x-y), 2)))