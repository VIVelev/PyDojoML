import numpy as np
from numpy import pi

from ..base import BaseModel
from ..exceptions import MethodNotSupportedError

__all__ = [
    "GaussianDist",
]

class GaussianDist(BaseModel):

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        return self

    def predict(self, X):
        X = np.array(X)
        assert len(X.shape) == 2

        return np.array([self.p(x) for x in X])

    def predict_proba(self, X):
        raise MethodNotSupportedError("""Probability predictions are not supported
                                        for Gaussian Distribution Anomaly Detection.""")

    def decision_function(self, X):
        raise MethodNotSupportedError("Decision function is not supported. Use `predict` instead.")

    def evaluate(self, X, y):
        raise MethodNotSupportedError("Evaluation methods are not available yet...")

    def p(self, x):
        assert type(x) is list or type(x) is np.ndarray

        return np.prod([
            1/(np.sqrt(2*pi) * self.std[j]) * \
            np.exp(-(x[j] - self.mean[j])**2 / (2*self.std[j]**2)) for j in range(x.size)
        ])
