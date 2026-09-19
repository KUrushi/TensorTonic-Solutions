import numpy as np

def linear_beta_schedule(T: int,
                         beta_1: float = 0.0001,
                         beta_T: float = 0.02) -> list[float]:
    """
    Returns T linearly spaced beta values.
    """
    if T == 1:
        return [beta_1]
        
    t = np.arange(0, T)
    beta_values = beta_1 + t * (beta_T - beta_1) / (T-1)
    return np.round(beta_values,6).tolist()