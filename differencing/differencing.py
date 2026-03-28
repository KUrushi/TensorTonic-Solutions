def _calc_differencing(series, order):
    if order == 0:
        return series
    else:
        output = []
        for i in range(1,len(series)):
            output.append(series[i]-series[i-1])     
        return _calc_differencing(output, order-1)
        
        
    
def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code here
    return _calc_differencing(series, order)