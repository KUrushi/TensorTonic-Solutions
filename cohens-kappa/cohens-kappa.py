import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    if len(rater1) == 0 or len(rater2) == 0:
        return 0.0
        
    r1_counter = {}
    r2_counter = {}
    agreements = 0
    for r1, r2 in zip(rater1, rater2) :
        if r1 == r2:
            agreements += 1
        if r1 not in r1_counter:
            r1_counter[r1] = 1
        else:
            r1_counter[r1] += 1
        if r2 not in r2_counter:
            r2_counter[r2] = 1
        else:
            r2_counter[r2] += 1

    po = agreements / len(rater1)
    pe = sum(r1_counter[k]/len(rater1) * r2_counter[k] / len(rater2) for k in r1_counter)
    if pe == 1:
        return 1.0
    return (po-pe)/(1-pe)