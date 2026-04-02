import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """
    # Your implementation here
    return -0.5 * np.sum(1 + 2 * log_var - np.power(mu, 2) - np.exp(log_var))
    