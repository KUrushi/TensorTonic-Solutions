import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    batch, seq_len, d_model = Q.shape
    d_head = d_model // num_heads
    Q = Q.reshape(batch, seq_len, num_heads, d_head).transpose(0,3,2,1)
    K = K.reshape(batch, seq_len, num_heads, d_head).transpose(0,3,2,1)
    V = V.reshape(batch, seq_len, num_heads, d_head).transpose(0,3,2,1)

    x = Q @ K.transpose(0,1,3,2)
    x = x / np.sqrt(d_head)
    x = softmax(x)
    x = x @ V    
    x = x.transpose(0,2,1,3)
    x = x.reshape(batch, seq_len, d_model)
    x = x @ W_o
    return x