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
        self.t = 1

    def update(self, W, dW):
        if self.v is None:
            self.v = np.zeros_like(W)

        # Exponentially Weighted Moving Average
        self.v = self.beta * self.v + (1 - self.beta) * dW
        # Bias correction
        # self.v /= (1 - self.beta**self.t)
        self.t += 1

        # Update
        return W - self.alpha * self.v

class RMSprop:
    pass

class Adam:
    pass
