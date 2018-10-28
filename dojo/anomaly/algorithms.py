import numpy as np
from scipy import linalg

from numpy import pi

from ..base import BaseModel
from ..exceptions import MethodNotSupportedError

__all__ = [
    "GaussianDist",
]


class GaussianDist(BaseModel):
    """Gaussian Distribution Anomaly Detection Algorithm
    
    Builds a Gaussion Distribution from the given data and it's
    features. Based on the feature's distributions the algorithm
    separates the anomalies from the normal examples.
    
    Parameters:
    -----------
    multi : boolean, whether to model Multivariate Gaussian Distribution
    or a Univariate one
    
    """

    def __init__(self, multi=False):
        self.multi = multi

        self.mean = None
        self.std = None
        self.sigma = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if self.multi:
            self.sigma = np.cov(np.transpose(X))

        return self

    def predict(self, X):
        X = super().predict(X)
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
        x = np.array(x)

        if not self.multi:
            return np.prod([
                1/(np.sqrt(2*pi) * self.std[j]) * \
                np.exp(-(x[j] - self.mean[j])**2 / (2*self.std[j]**2)) for j in range(x.size)
            ])

        else:
            return 1/(np.power(2*pi, x.size/2)*np.power(linalg.det(self.sigma), 1/2)) * \
            np.exp(-1/2 * (x - self.mean).T @ linalg.inv(self.sigma) @ (x-self.mean))
