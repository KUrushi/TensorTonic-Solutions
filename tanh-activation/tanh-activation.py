import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.array(x)
    exp_plus = np.exp(x)
    exp_minus = np.exp(-x)
    return (exp_plus - exp_minus) / (exp_plus + exp_minus)