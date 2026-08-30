import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    # Write code here
    outputs = {}
    keys = [k for k in train_dist.keys()]
    for k in keys:
        s = np.asarray(train_dist[k]) + eps
        t = np.asarray(serving_dist[k]) + eps

        psi = np.sum((s - t) * np.log(s / t))
        psi_rounded = round(float(psi), 6)
        is_skewed = psi >= threshold 
        outputs[k] = {"psi":psi_rounded, "skewed":bool(is_skewed)}
    return outputs