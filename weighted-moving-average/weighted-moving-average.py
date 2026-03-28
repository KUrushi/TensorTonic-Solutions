def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here
    sum_weight = sum(weights)
    k = len(weights)
    output = []
    for i in range(0, len(values)-len(weights)+1):
        x = values[i:i+k]
        value = sum([w * x_ for x_,w in zip(x,weights)]) / sum_weight
        output.append(value)

    return output
        
           
            