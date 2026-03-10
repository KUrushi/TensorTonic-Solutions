def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    Gs = [0 for _ in range(len(rewards))]
    G_mean = 0
    for i in range(-1, -len(rewards)-1, -1):
        if i == -1:
            Gt = rewards[i]
        else:
            Gt = rewards[i] + gamma * Gs[i+1]
        Gs[i] = Gt
        G_mean += Gt

    G_mean = G_mean / len(rewards)
    At = [g - G_mean for g in Gs]


    loss = 0
    for p, a in zip(log_probs, At):
        loss -= p * a
    return loss / len(log_probs)