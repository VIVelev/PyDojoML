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
    
    "precision",
    "recall",
    "f1_score",
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

def precision(y, y_pred):
    tp = true_positives(y, y_pred)
    fp = false_positives(y, y_pred)
    return  tp / (tp + fp)

def recall(y, y_pred):
    tp = true_positives(y, y_pred)
    fn = false_negatives(y, y_pred)
    return  tp / (tp + fn)

def f1_score(y, y_pred):
    p = precision(y, y_pred)
    r = recall(y, y_pred)

    return 2*p*r / (p+r)
