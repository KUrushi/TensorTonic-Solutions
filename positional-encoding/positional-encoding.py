import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)[:, np.newaxis]
    pos_dim = np.arange(d_model) // 2
    mask = np.arange(d_model) % 2
    value = pos/np.power(base,pos_dim* 2 / d_model)
    sin = np.sin(value)
    cos = np.cos(value)
    results = np.where(mask, cos, sin)
    return results