import numpy as np

def calc_value(P, R, gamma, V):
    return P * (R + gamma * V)

def bellman_optimality_backup(P, R, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    P = np.asarray(P)
    R = np.asarray(R)
    V = np.asarray(V)
    
    calculated_val = calc_value(P, R, gamma, V)
    next_state_value = np.sum(calculated_val, axis=2)
    v_new = np.max(next_state_value, axis=1)
    
    return np.round(v_new, 4).tolist()

