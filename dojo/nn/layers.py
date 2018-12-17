from abc import ABC, abstractmethod
import numpy as np

from ..activations import *
from ..exceptions import ParameterError
from ..regularizers import L2

__all__ = [
    "Layer",
    "Dense",
]


class Layer(ABC):

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, prev_A):
        pass

    @abstractmethod
    def backward(self, dA):
        pass

    @abstractmethod
    def update(self, alpha):
        pass

# ====================================================================================================
# ====================================================================================================

class Dense(Layer):
    # TODO: add __doc__

    def __init__(self, n_neurons, n_inputs=1, activation="sigmoid", regularizer=L2(0)):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.activation = activation
        self.regularizer = regularizer
        
        if activation == "sigmoid":
            self.activation_func = sigmoid
        elif activation == "tanh":
            self.activation_func = tanh
        elif activation == "relu":
            self.activation_func = relu
        elif activation == "leaky_relu":
            self.activation_func = leaky_relu
        else:
            raise ParameterError(f"Activation: \"{activation}\" not known.")

        self.W = None
        self.b = None
        self.init_weights()

        self.A_prev = None
        self.Z = None
        self.A = None

        self.grads = {}

    def init_weights(self):
        """Performs He initialization"""

        self.W = np.random.randn(self.n_neurons, self.n_inputs) * np.sqrt(2 / self.n_inputs)
        self.b = np.zeros((self.n_neurons, 1))

    def linear_forward(self):
        self.Z = self.W @ self.A_prev + self.b
        return self.Z

    def linear_activation_forward(self, A_prev):
        self.A_prev = A_prev
        self.A = self.activation_func(self.linear_forward())
        return self.A

    def forward(self, A_prev):
        return self.linear_activation_forward(A_prev)

    def linear_backward(self):
        m = self.A_prev.shape[1]
        self.grads["dW"] = 1/m * self.grads["dZ"] @ self.A_prev.T + 1/m * self.regularizer.gradient(self.W)
        self.grads["db"] = np.mean(self.grads["dZ"], axis=1, keepdims=True)
        self.grads["dA_prev"] = self.W.T @ self.grads["dZ"]

    def linear_activation_backward(self, dA):
        if self.activation == "sigmoid":
            self.grads["dZ"] = dA * self.A * (1 - self.A)

        elif self.activation == "tanh":
            self.grads["dZ"] = dA * (1 - self.A**2)

        elif self.activation == "relu":
            self.grads["dZ"] = dA * np.vectorize(lambda x: 1 if x >= 0 else 0)(self.A)

        else: # Leaky ReLU
            self.grads["dZ"] = dA * np.vectorize(lambda x: 1 if x >= 0 else 0.01)(self.A)

        self.linear_backward()

    def backward(self, dA):
        self.linear_activation_backward(dA)

    def update(self, alpha):
        self.W -= alpha*self.grads["dW"]
        self.b -= alpha*self.grads["db"]
