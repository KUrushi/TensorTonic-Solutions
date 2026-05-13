import numpy as np

def poisson_pmf(lam, k):
    k_factorial = np.prod(np.arange(1, k + 1))
    return np.exp(-lam) * np.power(lam, k) / k_factorial

def poisson_pmf_cdf(lam, k):
    cdf_value = 0.0
    for i in range(k + 1):
        pmf_i = poisson_pmf(lam, i)
        cdf_value += pmf_i
        
    pmf_k = poisson_pmf(lam, k)
    
    return pmf_k, cdf_value