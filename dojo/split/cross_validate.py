from ..metrics.regression import mean_squared_error
from .KFolds import KFolds

__all__ = [
    "cross_validate",
]


def cross_validate(model, X, y, cv=5, metric=mean_squared_error):
    """Cross Validation

    Evaluates the given model using the given data
    repetitively fitting and predicting on different
    chunks (folds) from the data.
    
    Parameters:
    -----------
    model : dojo-model, the model to be evaluated
    X : matrix, shape (n_samples, n_features), the data used for evaluation
    y : vector, shape (n_samples, ), the desired labels
    cv : integer, optional, the number of iterations
    metric : the single value error/accuracy metric
    
    Returns:
    --------
    dict_results : dictionary with train score and test score
    
    """

    # TODO: Add feature that handles the metric function
    # automatically when metric="auto".

    train_score = 0
    test_score = 0

    folds = KFolds(X, y, k=cv)

    for X_train, X_test, y_train, y_test in folds:
        model.fit(X_train, y_train)

        train_score += metric(y_train, model.predict(X_train))
        test_score += metric(y_test, model.predict(X_test))

    return {
        "train_score": train_score / folds.k,
        "test_score": test_score / folds.k
    }
