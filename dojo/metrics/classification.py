from .utils import (
    np, convert_assert,
    assert_binary_problem,
)

__all__ = [
    "accuracy_score",
    "accuracy",

    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
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

def true_positives(y, y_pred):
    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 1] == 1)

def false_positives(y, y_pred):
    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 0] == 1)

def true_negatives(y, y_pred):
    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 0] == 0)

def false_negatives(y, y_pred):
    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 1] == 0)
