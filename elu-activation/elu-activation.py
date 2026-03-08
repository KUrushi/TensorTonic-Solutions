def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    output = []
    for x_ in x:
       if x_ > 0:
           output.append(x_)
       else:
           output.append(alpha * (math.exp(x_) - 1))
    return output