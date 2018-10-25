import numpy as np
from numpy import pi

__all__ = [
    "GaussianDist",
]

class GaussianDist:
    def __init__(self):
        self.mean = []
        self.std = []

    def fit(self, X):
        self.mean = np.mean(X, axis=1)
        self.std = np.std(X, axis=1)

    def predict(self, X):
        return [self.p(x) for x in X]

    def p(self, x):
        return np.prod(
            1/(np.sqrt(2*pi) * self.std[j]) * \
            np.exp(-(x[j] - self.mean[j])**2 / (2*self.std[j]**2)) for j in range(x.size)
        )
