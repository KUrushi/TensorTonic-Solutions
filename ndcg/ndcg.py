import math
def dcg(rels:list[float], k:int)->float:
    k = min(len(rels), k)
    scores = 0
    for i in range(1, k+1):
        scores += (2**rels[i-1] - 1) / math.log2(i+1)
    return scores

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    if all([r == 0. for r in relevance_scores]):
        return 0.0
        
    return dcg(relevance_scores, k) / dcg(sorted(relevance_scores, reverse=True),k)