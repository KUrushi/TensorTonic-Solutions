import numpy as np
def relu(x):
    return np.where(x<0, 0, x)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    x = x @ W1 + b1
    x = relu(x)
    x = x @ W2 + b2
    return x