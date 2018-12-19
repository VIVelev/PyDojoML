from abc import abstractmethod
import numpy as np

__all__ = [
    "Optimizer",
    "GradientDescent",
    "Momentum",
    "RMSprop",
    "Adam",
]


class Optimizer:
    def __init__(self, alpha):
        self.alpha = alpha

    @abstractmethod
    def update(self, W, dW):
        pass

# ====================================================================================================
# ====================================================================================================

class GradientDescent(Optimizer):
    def __init__(self, alpha=0.01):
        super().__init__(alpha)

    def update(self, W, dW):
        return W - self.alpha * dW

class Momentum(Optimizer):
    def __init__(self, alpha=0.01, beta=0.9):
        super().__init__(alpha)
        self.beta = beta
        self.v = None

    def update(self, W, dW):
        if self.v is None:
            self.v = np.zeros_like(W)

        # Exponentially Weighted Moving Average
        self.v = self.beta * self.v + (1 - self.beta) * dW
        # Update
        return W - self.alpha * self.v

class RMSprop(Optimizer):
    def __init__(self, alpha=0.01, beta=0.999):
        super().__init__(alpha)
        self.beta = beta
        self.eps = 1e-8
        self.s = None

    def update(self, W, dW):
        if self.s is None:
            self.s = np.zeros_like(W)

        # Exponentially Weighted Moving Average
        self.s = self.beta * self.s + (1 - self.beta) * np.square(dW)
        # Update
        return W - self.alpha * dW / (np.sqrt(self.s) + self.eps)

class Adam:
    pass
