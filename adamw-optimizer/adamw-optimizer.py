import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    
    mt = beta1 * m + (1-beta1) * grad
    vt = beta2 * v + (1-beta2) * np.power(grad, 2)
    print(f"{m=}, {mt=}\n{v=}, {vt=}")
    wt = w - lr * (weight_decay * w) - lr * (mt / (np.sqrt(vt) + eps))
    
    return wt, mt, vt