import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)
    return_sum = np.zeros((n_states,))
    visit_counts = np.zeros((n_states,))

    for episode in episodes:
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            step = episode[t]
            state = step[0]
            reward = step[1]
            
            G = reward + gamma * G
            if state not in [s[0] for s in episode[:t]]:
                return_sum[state] += G
                visit_counts[state] += 1

    return np.divide(return_sum, visit_counts, out=np.zeros_like(return_sum), where=visit_counts!=0)