def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    L = (6 / (fan_in + fan_out)) ** 0.5
    return [[w * 2*L - L for w in W_] for W_ in W]