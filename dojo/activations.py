from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "Activation",
    "Sigmoid",
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

# ====================================================================================================
# ====================================================================================================

class Sigmoid(Activation):
    def __init__(self):
        pass

    def __call__(self, x):
        return np.vectorize(lambda k: 1.0/(1 + np.exp(-k)))(x)

    def gradient(self, x):
        a = self(x)
        return  a * (1 - a)

class TanH(Activation):
    def __init__(self):
        pass

    def __call__(self, x):
        return np.tanh(x)
    
    def gradient(self, x):
        return 1 - self(x)**2

class ReLU(Activation):
    def __init__(self):
        pass

    def __call__(self, x):
        return np.vectorize(max)(0, x)
    
    def gradient(self, x):
        return (np.array(x) >= 0).astype(int)

class LeakyReLU(Activation):
    def __init__(self, eps=0.01):
        self.eps = eps

    def __call__(self, x):
        return np.vectorize(max)(self.eps*x, x)

    def gradient(self, x):
        grad = (np.array(x) >= 0).astype(int)
        grad[grad == 0] = self.eps
        return grad
