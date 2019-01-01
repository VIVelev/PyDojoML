from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "Regularizer",
    "L1",
    "L2",
]


class Regularizer(ABC):
    @abstractmethod
    def __call__(self, W):
        pass

    @abstractmethod
    def gradient(self, W):
        pass

# ====================================================================================================
# ====================================================================================================

class L1(Regularizer):
    """Performs absolute norm"""

    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, W):
        return self.lambd * np.sum(np.abs(W))

    def gradient(self, W):
        return self.lambd * np.abs(W)

class L2(Regularizer):
    """Performs Forbenius Norm (squared norm)"""

    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, W):
        return (self.lambd / 2) * np.sum(np.square(W))

    def gradient(self, W):
        return self.lambd * W
