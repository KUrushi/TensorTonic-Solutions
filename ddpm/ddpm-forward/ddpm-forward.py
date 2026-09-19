import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    outputs = []
    value = 1.0
    for beta in betas:
        value *= 1-beta
        outputs.append(value)
    return outputs

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """
    outputs = []
    x_0 = np.asarray(x_0)
    epsilon = np.asarray(epsilon)
    alphas = np.asarray(get_alpha_bar(betas))
        
    for i in range(t):
        value = np.sqrt(alphas[i]) * x_0 + np.sqrt(1-alphas[i])*epsilon
        outputs.append(list(value))
    return outputs[-1]
        