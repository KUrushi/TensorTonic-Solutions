import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    mt = beta1 * m + (1-beta1) * grad
    vt = beta2 * v + (1-beta2) * np.power(grad, 2)
    mt_hat = mt / (1 - np.power(beta1, t))
    vt_hat = vt / (1 - np.power(beta2, t))
    new_param = param - lr * (mt_hat / (np.sqrt(vt_hat) + eps))
    return new_param, mt, vt