import numpy as np
from .utils import convert_assert

__all__ = [
    "accuracy_score",
    "accuracy",
]

def accuracy_score(y, y_pred):
    y, y_pred = convert_assert(y, y_pred)
    return np.count_nonzero(y == y_pred)/y.size

accuracy = accuracy_score
