from .utils import np
from ..base import BaseModel

from ..metrics.classification import accuracy_score
from .cost import cross_entropy

__all__ = [
    "NeuralNetwork",
]


class NeuralNetwork(BaseModel):
    # TODO: add __doc__

    def __init__(self, alpha=0.01, n_iterations=5_000):
        self.alpha = alpha
        self.n_iterations = n_iterations

        self._layers = []

    def add(self, layer):
        self._layers.append(layer)

    def forward(self, X):
        AL = X
        for layer in self._layers:
            AL = layer.linear_activation_forward(AL)

        return AL

    def backward(self):
        for layer in self._layers:
            layer.linear_activation_backward()

    def fit(self, X, y):
        for i in range(1, self.n_iterations + 1):
            AL = self.forward(X)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cross_entropy(y, AL)}")
            self.backward()

            for layer in self._layers:
                layer.update()

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        return self.forward(X)

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))
