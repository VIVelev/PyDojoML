from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "Optimizer",
    "Momentum",
    "RMSprop",
    "Adam",
]


class Optimizer(ABC):
    @abstractmethod
    def update(self, W, grad):
        pass

# ====================================================================================================
# ====================================================================================================

class Momentum:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.v = None
        self.t = 1

    def update(self, W, grad):
        if self.v is None:
            self.v = np.zeros_like(W)

        # Exponentially Weighted Moving Average
        self.v = self.beta * self.v + (1 - self.beta) * W
        # Bias correction
        self.v /= (1 - self.beta**self.t)
        self.t += 1

        # Update
        return W - self.alpha * self.v

class RMSprop:
    pass

class Adam:
    pass
