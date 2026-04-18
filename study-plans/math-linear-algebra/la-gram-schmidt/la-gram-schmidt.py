import numpy as np

def gram_schmidt(vectors):
    """
    Returns: float64 array of shape (k, n), orthonormal basis spanning the input space.
    """
    vectors = np.asarray(vectors)
    output = np.zeros(vectors.shape)
    v1 = vectors[0]
    v1_norm = np.linalg.norm(v1)
    if v1_norm != 0:
        output[0] = v1 / v1_norm
        
    for i in range(1, len(vectors)):
        v = vectors[i]
        q = output[:i]
        u = v - v @ q.T @ q

        u_norm = np.linalg.norm(u)
        if u_norm != 0:
            output[i] = u / u_norm
        
         
    return output