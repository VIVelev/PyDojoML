from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "Loss",
    "SquareLoss",
    "CrossEntropy",
]


class Loss(ABC):
    @abstractmethod
    def __call__(self, y, y_pred):
        pass

    @abstractmethod
    def gradient(self, y, y_pred):
        pass

# ==================================================================================================== #
# ==================================================================================================== #

class SquareLoss(Loss):
    def __init__(self):
        pass

    def __call__(self, y, y_pred):
        return .5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

# ==================================================================================================== #

class CrossEntropy(Loss):
    def __init__(self):
        pass

    def __call__(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.sum(- y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred), axis=0)

    def gradient(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y / y_pred) + (1 - y) / (1 - y_pred)

# ==================================================================================================== #
