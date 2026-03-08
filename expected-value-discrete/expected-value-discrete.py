import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)
    if not np.allclose(p.sum(), 1.0, rtol=1e-6):
        raise ValueError("Invalid Probability")
    
    return np.sum(x * p)
       
