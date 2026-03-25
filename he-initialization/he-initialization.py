def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    output = [[w for w in W_] for W_ in W]
    L = (6 / fan_in) ** 0.5
    for i in range(len(W)):
        for j in range(len(W[0])):
            output[i][j] = output[i][j]*2 * L - L
    return output
            