import numpy as np
from scipy import linalg

from ..base import BaseModel
from ..exceptions import MethodNotSupportedError
from ..metrics.regression import mean_squared_error

__all__ = [
    "LinearRegression",
]

class LinearRegression(BaseModel):
    def __init__(self, intercept=0, coefs=[], verbose=False):
        super().__init__()
        self.intercept = intercept
        self.coefs = coefs
        self.verbose = verbose

    def fit(self, X, y):
        super().fit(X, y)

        m, _ = X.shape
        X = np.hstack((
            np.array([[1] for _ in range(m)]),
            X
        ))

        self.intercept, *self.coefs = linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        super().predict(X)
        return [self.intercept + np.array(self.coefs).T @ x for x in X]
    
    def predict_proba(self, X):
        super().predict_proba(X)
        raise MethodNotSupportedError("Probability predictions are not supported for Linear Regression.")
    
    def decision_function(self, X):
        super().decision_function(X)
        raise MethodNotSupportedError("Use `predict` method instead.")

    def evaluate(self, X, y):
        super().evaluate(X, y)
        print(
            "Mean Squared Error: {}".format(mean_squared_error(y, self.predict(X)))
        )
