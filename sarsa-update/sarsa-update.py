def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    # Write code here
    output = [[a for a in s] for s in q_table]
    delta = reward + gamma * q_table[next_state][next_action] - q_table[state][action]
    output[state][action] += alpha * delta
    return output