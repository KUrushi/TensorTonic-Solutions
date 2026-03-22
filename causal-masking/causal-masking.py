import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    
    scores = np.array(scores)
    mask = np.tril(np.ones(scores.shape))
    mask = mask.astype(bool)
    return np.where(mask, scores, mask_value)
