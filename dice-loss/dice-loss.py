import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.array(p).flatten()
    y = np.array(y).flatten()
    return 1 - (2 * np.dot(p, y.T) + eps)/(np.sum(p) + np.sum(y) + eps)