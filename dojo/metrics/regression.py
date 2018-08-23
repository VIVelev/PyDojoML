from .utils import np, convert_assert

__all__ = [
    "squared_error",
    "mean_squared_error",
    "absolute_error",
    "mean_absolute_error",
]

def squared_error(y, y_pred):
    """Sum of the squared differences between target and prediction.
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target values.

    y_pred : vector, shape (n_samples,)
    The predicted values.
    
    Returns:
    --------
    error : float number
    
    """

    y, y_pred = convert_assert(y, y_pred)
    return np.sum((y - y_pred) ** 2)

def mean_squared_error(y, y_pred):
    """Mean squared difference between target and prediction.
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target values.

    y_pred : vector, shape (n_samples,)
    The predicted values.
    
    Returns:
    --------
    error : float number

    """

    return squared_error(y, y_pred) / len(y)

def absolute_error(y, y_pred):
    """Sum of the differences between target and prediction.
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target values.

    y_pred : vector, shape (n_samples,)
    The predicted values.
    
    Returns:
    --------
    error : float number
    
    """

    y, y_pred = convert_assert(y, y_pred)
    return np.sum(y - y_pred)

def mean_absolute_error(y, y_pred):
    """Mean difference between target and prediction.
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target values.

    y_pred : vector, shape (n_samples,)
    The predicted values.
    
    Returns:
    --------
    error : float number
    
    """

    return absolute_error(y, y_pred) / len(y)
