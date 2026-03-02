def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    # Write code here
    N = len(priorities)
    P = [p ** alpha for p in priorities]
    P_sum = sum(P)
    P = [p/P_sum for p in P]
    W = [(N * p) ** -beta for p in P]
    max_w = max(W)
    return [P,[w/max_w for w in W]]