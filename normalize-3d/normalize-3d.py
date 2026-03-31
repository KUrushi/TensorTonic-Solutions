import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    norm = np.linalg.norm(v,axis=-1,keepdims=True)

    norm[norm==0] = 1
    return v / norm
