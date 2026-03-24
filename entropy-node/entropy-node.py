import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0
    # Write code here
    counter = {
        
    }
    for i in y:
        if i not in counter:
            counter[i] = 1
        else:
            counter[i] += 1


    hs = 0
    for v in counter.values():
        prob = v / len(y)
        hs -= (prob * np.log2(prob))

    return hs
        