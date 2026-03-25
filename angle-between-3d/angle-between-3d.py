import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.array(v)
    w = np.array(w)
    v_norm =  np.linalg.norm(v, ord=2)
    w_norm = np.linalg.norm(w, ord=2)

    if v_norm == 0.0 or w_norm == 0.0:
        return np.nan
        
    dot = np.dot(v, w.T)
    cos = dot / (v_norm * w_norm)
    return np.arccos(cos)