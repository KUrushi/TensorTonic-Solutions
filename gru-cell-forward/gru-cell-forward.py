import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    x = np.array(x)
    h_prev = np.array(h_prev)
    params = {
        k: np.array(v)
        for k, v in params.items()
    }
    
    zt = _sigmoid(x @ params['Wz'] + h_prev @ params['Uz'] + params['bz'])
    rt = _sigmoid(x @ params['Wr'] + h_prev @ params['Ur'] + params['br'])
    
    ht = np.tanh(x @ params['Wh'] + (rt * h_prev) @ params['Uh'] + params['bh'])
    
    return (1 - zt) * h_prev + zt * ht