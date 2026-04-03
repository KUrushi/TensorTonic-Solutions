import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    h_t = h_0
    batch_size, seq_len, dims = X.shape
    outputs = np.zeros((batch_size, seq_len, b_h.shape[-1]))
    
    for i in range(seq_len):
        h_t = np.tanh(X[:,i,:] @ W_xh.T + h_t @ W_hh.T + b_h)
        outputs[:,i,:] = h_t
    return np.array(outputs), h_t
        