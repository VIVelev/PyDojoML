import numpy as np
from scipy import linalg
# from numba import jit

from ..base import BaseModel

__all__ = [
    "LinearRegression",
]

class LinearRegression(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)
        self.intercept = 0
        self.coefs = []

        self.set_params(intercept=self.intercept, coefs=self.coefs)

    def fit(self, X, y):
        if type(X) is not np.ndarray:
            X = np.array(X)
        
        if type(y) is not np.ndarray:
            y = np.array(y)

        m, _ = X.shape
        X = np.hstack((
            np.array([[1] for _ in range(m)]),
            X
        ))

        self.intercept, *self.coefs = linalg.inv(X.T @ X) @ X.T @ y
        self.set_params(intercept=self.intercept, coefs=self.coefs)
        return self

    def predict(self, X):
        return [self.intercept + self.coefs @ X[i, :] for i in range(X.shape[0])]

    def predict_proba(self, X):
        raise Exception
    
    def decision_function(self, X):
        raise Exception

    def evaluate(self, X, y):
        pass
