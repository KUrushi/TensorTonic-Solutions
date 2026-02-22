def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    if len(ground_truth) == 0:
        return 0.0
    total = 0
    for recommend, g in zip(recommendations, ground_truth):
        top_k = recommend[:k]
        total += len(set(top_k) & set(g))
    return total / len(ground_truth)