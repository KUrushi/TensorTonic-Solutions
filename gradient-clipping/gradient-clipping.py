import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g_array = np.array(g)
    
    if max_norm <= 0:
        return g_array.copy()
        
    g_norm = np.linalg.norm(g_array)
    if g_norm == 0 or g_norm <= max_norm:
        return g_array.copy()
        
    return g_array * (max_norm / g_norm)
