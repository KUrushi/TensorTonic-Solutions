def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    max_values = max(values)
    min_values = min(values)

    w = (max_values - min_values) / num_bins

    return [min((x - min_values)//w, num_bins-1) if w != 0 else 0 for x in values ]

    