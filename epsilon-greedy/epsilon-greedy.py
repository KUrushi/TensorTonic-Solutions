import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    num_actions = len(q_values)

    if rng is not None:
        random_val = rng.random()
    else:
        random_val = np.random.rand()

    if random_val < epsilon:
        if rng is not None:
            action = rng.integers(0, num_actions)
        else:
            action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(q_values)
        
    return int(action)