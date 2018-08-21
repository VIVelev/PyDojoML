import numpy as np
from scipy.optimize import minimize

from ..base import BaseModel

__all__ = [
    "sigmoid",
    "LogisticRegression",
]

def sigmoid(x):
    z = lambda k: 1.0/(1 + np.exp(-k))

    if type(x) in [list, np.ndarray]:
        for i in range(len(x)):
            x[i] = z(x[i])
        return x

    else:
        return z(x)

class LogisticRegression(BaseModel):
    def __init__(self, intercept=0, coefs=[], verbose=False):
        self.intercept = intercept
        self.coefs = coefs
        self.verbose = verbose

        self._X, self._y = [], []

    def _loss(self, fittable):
        h = lambda x: sigmoid(fittable[0] + fittable[1:] @ x)
        y_pred = [1-(1e-18) if h(x) == 1 else 1e-18 for x in self._X]
        m, _ = self._X.shape

        return -(1.0/m) * sum(
            self._y[i]*np.log(y_pred[i]) + (1.0-self._y[i])*np.log(1.0-y_pred[i]) for i in range(m)
        )

    def _gradient(self, fittable):
        pass

    def fit(self, X, y):
        self._X, self._y = super().fit(X, y)

        res = minimize(
            self._loss,
            np.random.rand(1+self._X.shape[1]),
            method="BFGS",
            jac=self._gradient,
            options= {
                "maxiter": 500,
                "disp": self.verbose
            }
        )

        self.intercept, self.coefs = res.x[0], list(res.x[1:])
        return self

    def predict(self, X):
        return list(
            np.round(self.predict_proba(X))
        )

    def predict_proba(self, X):
        return list(
            sigmoid(self.decision_function(X))
        )

    def decision_function(self, X):
        X = super().decision_function(X)
        return [self.intercept + np.array(self.coefs).T @ x for x in X]

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        # TODO: implement
        raise NotImplementedError("This method is not yet implemented.")
