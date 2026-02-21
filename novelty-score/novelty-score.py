
def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Write code here
    scores = 0
    for i in recommendations:
        if i < len(recommendations):
            scores += math.log2(item_counts[i] / n_users)
    scores = -scores
    return scores /  len((recommendations))
        