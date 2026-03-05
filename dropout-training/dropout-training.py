import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.array(x)
    
    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)
        
    keep_mask = random_values < (1.0 - p)
    dropout_pattern = keep_mask * (1.0 / (1.0 - p))
    
    output = x * dropout_pattern
    
    return output, dropout_pattern