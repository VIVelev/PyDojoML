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
    """True-positives
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    tp : integer, the number of true-positives
    
    """

    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 1] == 1)

def false_positives(y, y_pred):
    """False-positives
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    fp : integer, the number of false-positives
    
    """

    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 0] == 1)

def true_negatives(y, y_pred):
    """True-negatives
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    tn : integer, the number of true-negatives
    
    """

    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 0] == 0)

def false_negatives(y, y_pred):
    """False-negatives
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    fn : integer, the number of false-negatives
    
    """

    y, y_pred = convert_assert(y, y_pred)
    assert_binary_problem(y)

    return np.count_nonzero(y_pred[y == 1] == 0)

def precision(y, y_pred):
    """Precision score

    precision = true_positives / (true_positives + false_positives)
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    precision : float
    
    """

    tp = true_positives(y, y_pred)
    fp = false_positives(y, y_pred)
    return  tp / (tp + fp)

def recall(y, y_pred):
    """Recall score

    recall = true_positives / (true_positives + false_negatives)
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    recall : float
    
    """

    tp = true_positives(y, y_pred)
    fn = false_negatives(y, y_pred)
    return  tp / (tp + fn)

def f1_score(y, y_pred):
    """F1 score

    f1_score = 2 * precision*recall / (precision + recall)
    
    Parameters:
    -----------
    y : vector, shape (n_samples,)
    The target labels.

    y_pred : vector, shape (n_samples,)
    The predicted labels.
    
    Returns:
    --------
    f1_score : float
    
    """

    p = precision(y, y_pred)
    r = recall(y, y_pred)

    return 2*p*r / (p+r)
