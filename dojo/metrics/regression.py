from .utils import np, convert_assert

__all__ = [
    "squared_error",
    "mean_squared_error",
    "mean_absolute_error",
]

def squared_error(y, y_pred):
    y, y_pred = convert_assert(y, y_pred)
    return np.sum((y - y_pred) ** 2)

def mean_squared_error(y, y_pred):
    return squared_error(y, y_pred)/y.size

def mean_absolute_error(y, y_pred):
    y, y_pred = convert_assert(y, y_pred)
    return np.sum(y - y_pred)/y.size
