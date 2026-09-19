import numpy as np

def compute_ddpm_loss(epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean DDPM noise-prediction loss.
    """
    epsilon = np.asarray(epsilon)
    epsilon_pred = np.asarray(epsilon_pred)
    return np.mean(np.power(epsilon-epsilon_pred,2))