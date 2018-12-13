import numpy as np
from ..base import BaseModel

from ..metrics.classification import accuracy_score
from .cost import cross_entropy

__all__ = [
    "NeuralNetwork",
]


class NeuralNetwork(BaseModel):
    # TODO: add __doc__

    def __init__(self, alpha=0.01, n_iterations=5_000, verbose=False):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.verbose = verbose

        self.last_cost = 0
        self._layers = []

    def add(self, layer):
        self._layers.append(layer)

    def forward(self, X):
        AL = X
        for layer in self._layers:
            AL = layer.linear_activation_forward(AL)

        return AL

    def backward(self, Y, AL):
        # Cross Entropy dA
        dA = - Y / AL + (1 - Y) / (1 - AL)

        # Back-propagation
        for layer in reversed(self._layers):
            layer.linear_activation_backward(dA)
            dA = layer.grads["dA_prev"]

    def fit(self, X, y):
        for i in range(1, self.n_iterations + 1):
            AL = self.forward(X)
            self.last_cost = cross_entropy(y, AL)
            if i % 100 == 0 and self.verbose:
                print(f"Iteration {i}, Cost: {self.last_cost}")
            self.backward(y, AL)

            for layer in self._layers:
                layer.update(self.alpha)

        return self

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        return self.forward(X)

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))
