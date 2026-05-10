def discounted_returns(rewards, gamma):
    """
    Returns: list of G_t values, one per timestep, each rounded to 4 decimals
    """
    Gt = 0.0
    outputs = []
    
    for r in reversed(rewards):
        Gt = r + gamma * Gt
        
        outputs.append(round(Gt, 4))

    return outputs[::-1]