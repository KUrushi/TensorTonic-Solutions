def norm(xs):
    output = 0
    for x in xs:
        output += x ** 2
    return output ** 0.5

def cosine(x1, x2):
    x1_norm = norm(x1)
    x2_norm = norm(x2)
    cos = 0
    for x1_, x2_ in zip(x1,x2):
        cos += x1_ * x2_

    return cos / (x1_norm * x2_norm)
        
        
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    cos = cosine(x1, x2)
    if label == 1:
        return 1 - cos
    elif label == -1:
        loss = cos - margin 
        return max(0, loss)
    else:
        raise ValueError(f"label is invalid: {label}")