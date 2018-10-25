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
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        return self

    def predict(self, X):
        return np.array([self.p(x) for x in np.array(X)])

    def p(self, x):
        return np.prod([
            1/(np.sqrt(2*pi) * self.std[j]) * \
            np.exp(-(x[j] - self.mean[j])**2 / (2*self.std[j]**2)) for j in range(x.size)
        ])
