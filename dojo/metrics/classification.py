from .utils import np, convert_assert

__all__ = [
    "accuracy_score",
    "accuracy",
]

def accuracy_score(y, y_pred):
    """Calculates the fraction of the correctly
    classified samples over all.
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    accuracy : float number, the fraction
    of the correctly classified samples over all
    
    """

    y, y_pred = convert_assert(y, y_pred)
    return np.count_nonzero(y == y_pred) / y.size

accuracy = accuracy_score
