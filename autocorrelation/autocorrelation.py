def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    if max_lag == 0:
        return [1.0]
    mean = sum(series) / len(series)
    variance = sum((x - mean)**2 for x in series)
    results = [1.0]
    if variance == 0:
        return results + [0.0 for k in range(1, max_lag+1)]
    for k in range(1, max_lag+1):
        total = 0
        for t in range(0, len(series)-k):
            total += (series[t]-mean) * (series[t+k]-mean)
        cor = total / variance
        results.append(cor)
    return results
        