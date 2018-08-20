import numpy as np
from scipy import linalg

from ..base import BaseModel

__all__ = [
    "LinearRegression",
]

class LinearRegression(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)
        self.set_params(intercept=0, coefs=[])

    def fit(self, X, y):
        super().fit(X, y)

        m, _ = X.shape
        X = np.hstack((
            np.array([[1] for _ in range(m)]),
            X
        ))

        res = linalg.inv(X.T @ X) @ X.T @ y
        self.set_params(intercept=res[0], coefs=res[1:])
        return self

    def predict(self, X):
        super().predict(X)

        intercept, coefs = self.get_params("intercept", "coefs")
        return [intercept + coefs.T @ X[i, :] for i in range(X.shape[0])]
    
    def predict_proba(self, X):
        super().predict_proba(X)

        raise Exception
    
    def decision_function(self, X):
        super().decision_function(X)

        raise Exception

    def evaluate(self, X, y):
        super().evaluate(X, y)

        raise Exception
