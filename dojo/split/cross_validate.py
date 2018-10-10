from ..metrics.regression import mean_squared_error
from ..metrics.classification import accuracy_score

from .KFolds import KFolds

__all__ = [
    "cross_validate",
]

def cross_validate(model, X, y, cv=5, metric="auto"):
    if metric == "auto":
        metric = mean_squared_error
    
    train_score = 0
    test_score = 0

    folds = KFolds(X, y, k=cv)

    for X_train, X_test, y_train, y_test in folds:
        model.fit(X_train, y_train)

        train_score += metric(y_train, model.predict(X_train))
        test_score += metric(y_test, model.predict(X_test))

    return train_score / folds.k, test_score / folds.k
