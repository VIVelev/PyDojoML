from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "Activation",
    "Linear",
    "Sigmoid",
    "Softmax",
    "TanH",
    "ReLU",
    "LeakyReLU",
]


class Activation(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

# ==================================================================================================== #
# ==================================================================================================== #

class Linear(Activation):
    """Computing the Linear (Identity) function and its derivative.

        g(x) = x
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1

# ==================================================================================================== #

class Sigmoid(Activation):
    """Computing the Sigmoid function and its derivative.

        g(x) = 1 / (1 + e^-x)
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return np.vectorize(lambda k: 1.0/(1 + np.exp(-k)))(x)

    def gradient(self, x):
        a = self(x)
        return a * (1 - a)

# ==================================================================================================== #

class Softmax(Activation):
    """Computing the numerical stable Softmax function and its derivative.

        g(vec) = e^vec[j] / sum(e^vec[k])
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        return exps / np.sum(exps, axis=self.axis, keepdims=True)

    def gradient(self, x):
        a = self(x)
        return a * (1 - a)

# ==================================================================================================== #

class TanH(Activation):
    """Computing the TanH function and its derivative.

        g(x) = (e^x - e^-x) / (e^x + e^-x)
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return np.tanh(x)
    
    def gradient(self, x):
        return 1 - self(x)**2

# ==================================================================================================== #

class ReLU(Activation):
    """Computing the ReLU function and its derivative.

        g(x) = max(x, 0)
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return np.vectorize(max)(0, x)
    
    def gradient(self, x):
        return (np.array(x) >= 0).astype(int)

# ==================================================================================================== #

class LeakyReLU(Activation):
    """Computing the ReLU function and its derivative.

        g(x) = max(x, eps*x)
    """

    def __init__(self, eps=0.01):
        self.eps = eps

    def __call__(self, x):
        return np.vectorize(max)(self.eps*x, x)

    def gradient(self, x):
        grad = (np.array(x) >= 0).astype(int)
        grad[grad == 0] = self.eps
        return grad

# ==================================================================================================== #
