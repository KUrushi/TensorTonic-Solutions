def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    loss = 0
    K = len(predictions)
    for i, p in enumerate(predictions):
        if i == target:
            qi = (1-epsilon) + (epsilon / K)
        elif i != target:
            qi = epsilon / K

        loss -= qi * math.log(p)
    return loss