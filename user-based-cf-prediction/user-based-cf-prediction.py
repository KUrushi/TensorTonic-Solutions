def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    # Write code here
    numerator = 0.0
    denominator = 0.0
    
    for s_u, r_u in zip(similarities, ratings):
        if s_u <= 0.0:
            continue
        numerator += s_u * r_u
        denominator += s_u
    if denominator == 0:
        return 0.0

    return numerator / denominator