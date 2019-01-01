from abc import ABC, abstractmethod
import copy
import numpy as np

from ..activations import *
from ..exceptions import ParameterError
from ..regularizers import L2

__all__ = [
    "Layer",
    "Dense",
    "ActivationLayer",
]


class Layer(ABC):

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_n_params(self):
        pass

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
    def update(self, optimizer):
        pass

# ====================================================================================================
# ====================================================================================================

class Dense(Layer):
    # TODO: add __doc__

    def __init__(self, n_neurons, n_inputs=1, activation="linear", regularizer=L2(0)):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        if activation.lower() == "linear":
            self.activation_func = Linear()
        elif activation.lower() == "sigmoid":
            self.activation_func = Sigmoid()
        elif activation.lower() == "softmax":
            self.activation_func = Softmax(axis=0)
        elif activation.lower() == "tanh":
            self.activation_func = TanH()
        elif activation.lower() == "relu":
            self.activation_func = ReLU()
        elif activation.lower() == "leaky_relu":
            self.activation_func = LeakyReLU()
        elif isinstance(activation, Activation):
            self.activation_func = activation
        else:
            raise ParameterError(f"Unknown activation function: `{activation}`")

        self.regularizer = regularizer

        self.W = None
        self.b = None
        self.init_weights()    
        self.W_opt = None
        self.b_opt = None

        self.A_prev = None
        self.Z = None
        self.A = None

        self.grads = {}

    def get_name(self):
        return "Dense"

    def get_n_params(self):
        return self.W.size + self.b.size

    def init_weights(self):
        """Performs He initialization"""

        self.W = np.random.randn(self.n_neurons, self.n_inputs) * np.sqrt(2 / self.n_inputs)
        self.b = np.zeros((self.n_neurons, 1))

    def linear_forward(self):
        self.Z = self.W @ self.A_prev + self.b

    def linear_activation_forward(self, A_prev):
        self.A_prev = A_prev
        self.linear_forward()
        self.A = self.activation_func(self.Z)

    def forward(self, A_prev):
        self.linear_activation_forward(A_prev)
        return self.A

    def linear_backward(self):
        m = self.A_prev.shape[1]
        self.grads["dW"] = 1/m * self.grads["dZ"] @ self.A_prev.T + 1/m * self.regularizer.gradient(self.W)
        self.grads["db"] = np.mean(self.grads["dZ"], axis=1, keepdims=True)
        self.grads["dA_prev"] = self.W.T @ self.grads["dZ"]

    def linear_activation_backward(self, dA):
        self.grads["dZ"] = dA * self.activation_func.gradient(self.Z)
        self.linear_backward()

    def backward(self, dA):
        self.linear_activation_backward(dA)
        return self.grads["dA_prev"]

    def update(self, optimizer):
        if self.W_opt is None and self.b_opt is None:
            self.W_opt = copy.copy(optimizer)
            self.b_opt = copy.copy(optimizer)

        self.W = self.W_opt.update(self.W, self.grads["dW"])
        self.b = self.b_opt.update(self.b, self.grads["db"])

class ActivationLayer(Layer):
    # TODO: add __doc__

    def __init__(self, activation):
        if activation.lower() == "linear":
            self.activation_func = Linear()
        elif activation.lower() == "sigmoid":
            self.activation_func = Sigmoid()
        elif activation.lower() == "softmax":
            self.activation_func = Softmax(axis=0)
        elif activation.lower() == "tanh":
            self.activation_func = TanH()
        elif activation.lower() == "relu":
            self.activation_func = ReLU()
        elif activation.lower() == "leaky_relu":
            self.activation_func = LeakyReLU()
        elif isinstance(activation, Activation):
            self.activation_func = activation
        else:
            raise ParameterError(f"Unknown activation function: `{activation}`")

        self.n_inputs = None
        self.n_neurons = None

        self.Z = None
        self.A = None
        self.dZ = None
        self.dA = None

    def get_name(self):
        return "Activation"

    def get_n_params(self):
        return 0

    def init_weights(self):
        pass

    def forward(self, Z):
        self.Z = Z
        self.A = self.activation_func(self.Z)
        return self.A

    def backward(self, dA):
        self.dA = dA
        self.dZ = self.dA * self.activation_func.gradient(self.Z)
        return self.dZ

    def update(self, optimizer):
        pass
