import numpy as np

def distance_sq(x, y):
    return np.sum(np.square(x - y), axis=1)

def triplet_loss(anchor, positive, negative, margin=1.0):
    anchor = np.atleast_2d(anchor)
    positive = np.atleast_2d(positive)
    negative = np.atleast_2d(negative)
    
    d_pos = distance_sq(anchor, positive)
    d_neg = distance_sq(anchor, negative)
    
    losses = np.maximum(0, d_pos - d_neg + margin)
    result = np.mean(losses)
    
    return float(result)