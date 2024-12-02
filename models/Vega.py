import numpy as np
from scipy.stats import norm

def Vega(S, K, T, R, Sigma) -> float:
    d1 = (np.log(S/K) + (R + 0.5 * Sigma ** 2) * T)/(Sigma * np.sqrt(T))
    vega = S* norm.pdf(d1) * np.sqrt(T)
    return vega