import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    output = np.asarray(X)
    match strategy:
        case "mean":
            fill_value = np.nanmean(output, axis=0)
        case "median":
            fill_value = np.nanmedian(output, axis=0)
    fill_value = np.where(np.isnan(fill_value), 0, fill_value)
    output = np.where(np.isnan(output), fill_value, output)

    return output