from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "Loss",
    "SquareLoss",
    "CrossEntropy",
    "KL_Divergence",
]


class Loss(ABC):
    @abstractmethod
    def __call__(self, y, y_pred):
        pass

    @abstractmethod
    def gradient(self, y, y_pred):
        pass

# ==================================================================================================== #
# ==================================================================================================== #

class SquareLoss(Loss):
    def __init__(self):
        pass

    def __call__(self, y, y_pred):
        return .5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

# ==================================================================================================== #

class CrossEntropy(Loss):
    def __init__(self):
        pass

    def __call__(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.sum(- y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred), axis=0)

    def gradient(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y / y_pred) + (1 - y) / (1 - y_pred)

# ==================================================================================================== #

class KL_Divergence(Loss):
    """Kullback–Leibler divergence (KL divergence)

    A measure of how one probability distribution is different from a second, reference probability distribution.
    Wikipedia page: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    
    The one probability distribution is Gaussian Distribution.
    The reference probability distribution is Student t-Distribution with degree of freedom 1.

    """

    def __init__(self):
        pass

    def __call__(self, P, Q):
        cost = 0
        for i in range(P.shape[0]):
            for j in range(Q.shape[0]):
                if P[i, j] != 0 and Q[i, j] != 0:
                    cost += P[i, j] * np.log( P[i, j] / Q[i, j] )
            
        return cost

    def gradient(self, P, Q, Y, i):
        """Computes the gradient of KL divergence with respect to the i'th example of Y"""

        return 4 * sum([
            (P[i, j] - Q[i, j]) * (Y[i] - Y[j]) * (1 + np.linalg.norm(Y[i] - Y[j]) ** 2) ** -1 \
            for j in range(Y.shape[0])
        ])

# ==================================================================================================== #
