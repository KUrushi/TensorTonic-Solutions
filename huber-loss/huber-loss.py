import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    return np.mean(np.where(np.abs(diff)<=delta, np.power(diff, 2)/2, delta * (np.abs(diff) - delta / 2)))