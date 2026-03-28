def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    # Write code here
    rolling_mean = []
    rolling_sigma = [] 
    for i in range(len(values) - window_size+1):
        mean = sum(values[i:i+window_size]) / window_size
        sigma = (sum([(v-mean) ** 2 for v in values[i:i+window_size]]) / window_size) ** 0.5
        rolling_sigma.append(sigma)

    return rolling_sigma
    