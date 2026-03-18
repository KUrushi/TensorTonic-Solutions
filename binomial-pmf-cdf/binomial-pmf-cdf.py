import numpy as np
from scipy.special import comb

def calc_pmf(n,p,k):
    return comb(n,k) * np.power(p, k) * np.power(1-p, n-k)

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    pmf = 0
    cdf = 0
    for i in range(k+1):
        prob = calc_pmf(n,p,i)
        if i == k:
            pmf = prob
        cdf += prob

    return (pmf, cdf)
        