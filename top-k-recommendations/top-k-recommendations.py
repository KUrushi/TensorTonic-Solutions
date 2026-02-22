def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    new_scores = []
    for i in range(len(scores)):
        if i not in (rated_indices):
            new_scores.append((i, scores[i]))

    return [i for i, _ in sorted(new_scores, key=lambda x: x[1],reverse=True)[:k]]