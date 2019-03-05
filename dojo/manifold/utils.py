import numpy as np
from ..base import Preprocessor

__all__ = [
    'kl_divergence',
]


def kl_divergence(P, Q):
    """Kullback–Leibler divergence (KL divergence)

    A measure of how one probability distribution is different from a second, reference probability distribution.
    Wikipedia page: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    """

    cost = 0
    for i in range(P.shape[0]):
        for j in range(Q.shape[0]):
            if P[i, j] != 0 and Q[i, j] != 0:
                cost += P[i, j] * np.log( P[i, j] / Q[i, j] )
        
    return cost
