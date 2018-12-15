import numpy as np
from ..base import BaseModel

from ..metrics.classification import accuracy_score
from .losses import CrossEntropy

__all__ = [
    "NeuralNetwork",
]


class NeuralNetwork(BaseModel):
    # TODO: add __doc__

    def __init__(self, alpha=0.01, n_iterations=5_000, loss=CrossEntropy(), verbose=False):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.loss = loss
        self.verbose = verbose

        self._loss_values = []
        self._layers = []

    def add(self, layer):
        self._layers.append(layer)

    def forwardprop(self, X):
        AL = X
        for layer in self._layers:
            AL = layer.forward(AL)

        return AL

    def backprop(self, Y, AL):
        dA = self.loss.gradient(Y, AL)

        for layer in reversed(self._layers):
            layer.backward(dA)
            dA = layer.grads["dA_prev"]

    def fit(self, X, y):
        X, y = super().fit(X, y)
        X = X.T

        for i in range(1, self.n_iterations + 1):
            AL = self.forwardprop(X)
            self._loss_values.append(np.mean(self.loss(y, AL)))
            if i % 100 == 0 and self.verbose:
                print(f"Iteration {i}, Cost: {self._loss_values[-1]}")
            self.backprop(y, AL)

            for layer in self._layers:
                layer.update(self.alpha)

        return self

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        X = super().predict_proba(X).T
        return self.forwardprop(X)

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))
