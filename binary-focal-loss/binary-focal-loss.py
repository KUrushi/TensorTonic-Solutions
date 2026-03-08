def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    loss = 0
    for p, t in zip(predictions, targets):
        if t == 1:
            pt = p
        elif t == 0:
            pt = 1-p
        else:
            raise ValueError(f"Invalid target: {t}")

        loss += -alpha * ((1 - pt) ** gamma) * math.log(pt)
    return loss / len(predictions)