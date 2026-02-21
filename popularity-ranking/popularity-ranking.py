def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    scores = []
    for item_ratings, vote_counts in items:
        score = vote_counts / (vote_counts + min_votes) * item_ratings + (min_votes) / (vote_counts + min_votes) * global_mean
        scores.append(score)
    return scores