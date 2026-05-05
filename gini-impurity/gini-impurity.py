import numpy as np

def compute_gini(ps):
    return 1 - sum(p ** 2 for p in ps)

def calc_probs(labels):
    counter = {}
    for label in labels:
        if label not in counter:
            counter[label] = 1
        else:
            counter[label] += 1
    return [v / len(labels) for v in counter.values()]
    
def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    Nl = len(y_left)
    Nr = len(y_right)
    N = Nl + Nr
    if N == 0:
        return 0.0 
    gini_l = compute_gini(calc_probs(y_left))
    gini_r = compute_gini(calc_probs(y_right))
    
    gini = 0.0
    if gini_l != 0.0:
        gini += Nl/N * gini_l
        
    if gini_r != 0.0:
        gini += Nr/N * gini_r
        
    return  gini