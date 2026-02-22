import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length)[:, np.newaxis]
    dim = np.arange(d_model)

    angle_rads = pos / np.power(10000, (2 * (dim // 2))/d_model)
    pe = np.zeros((seq_length, d_model))

    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pe