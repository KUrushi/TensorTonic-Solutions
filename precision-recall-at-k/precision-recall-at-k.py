def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    hit = len(set(recommended[:k]) & set(relevant))
    precision = hit / k
    recall = hit / len(relevant)
    return [precision, recall]