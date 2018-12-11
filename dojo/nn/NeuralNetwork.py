import numpy as np
from ..base import BaseModel

from ..metrics.classification import accuracy_score

__all__ = [
    "NeuralNetwork",
]


class NeuralNetwork(BaseModel):
    # TODO: add __doc__

    def __init__(self, alpha=0.01, layers_dims=[1, 1]):
        self.alpha = alpha
        self.layers_dims = layers_dims
        self.L = len(layers_dims)-1

        self._parameters = {}
        self._grads = {}

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))
