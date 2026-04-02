import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    # Your implementation here
    std = np.power(np.e, log_var * 0.5)
    return mu + std * np.random.normal(0,1,size=mu.shape)