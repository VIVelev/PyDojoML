import numpy as np

__all__ = [
    "squared_error",
    "mean_squared_error",
]

def squared_error(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    assert y.size == y_pred.size

    return np.sum((y - y_pred) ** 2)

def mean_squared_error(y, y_pred):
    return squared_error(y, y_pred)/y.size
