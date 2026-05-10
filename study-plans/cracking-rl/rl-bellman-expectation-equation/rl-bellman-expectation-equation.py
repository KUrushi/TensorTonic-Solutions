def bellman_expectation_backup(P, R, policy, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    V_new = [0.0 for _ in V]
    for state_index in range(len(P)):
        new_state_value = 0.0
        for action_index in range(len(P[state_index])):
            next_state_value = 0.0
            for next_state_index in range(len(P[state_index][action_index])):
                next_state_value += P[state_index][action_index][next_state_index] * (R[state_index][action_index][next_state_index] + gamma * V[next_state_index])

            new_state_value += policy[state_index][action_index] * next_state_value
        V_new[state_index] = new_state_value

    return [round(v, 4) for v in V_new]