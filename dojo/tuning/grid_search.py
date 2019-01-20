import numpy as np
from copy import copy
from ..base import Classifier, Regressor
from ..split import cross_validate

__all__ = [
    "GridSearch",
]


class GridSearch(Classifier, Regressor):
    """Exhaustive search over specified parameter values for a model.

    Parameters:
    -----------
    model : Dojo-Model
    param_grid : dict
    Dictionary with parameters names (string) as keys and lists of parameter settings to try as values
    cv_folds : integer, optional, the number of iterations/folds for cross-validation
    metric : the single value error/accuracy metric, optional

    """

    def __init__(self, model, param_grid={}, cv_folds=5, metric="auto"):
        self.model = model
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.metric = metric

        self.best_model = None
        self.best_score = 10**9 if isinstance(self.model, Regressor) else 0

    def fit(self, X, y):
        n_params = len(self.param_grid)
        idxs = [0 for _ in range(n_params)]
        lens = [len(list(self.param_grid.values())[i]) for i in range(n_params)]
        iter_idx = starting_iter_idx = n_params-1

        while iter_idx != -1:
            current_values = [list(self.param_grid.values())[i][idxs[i]] for i in range(n_params)]
            params = dict(zip(self.param_grid.keys(), current_values))

            current_model = copy(self.model)
            current_model.set_params(**params)

            current_score = cross_validate(current_model, X, y, k_folds=self.cv_folds, metric=self.metric)["test_scores"].mean()

            if ((isinstance(self.model, Regressor) and current_score < self.best_score) or
                (not isinstance(self.model, Regressor) and current_score > self.best_score)):
                self.best_model = current_model            
                self.best_score = current_score

            if iter_idx == n_params:
                    idxs = [0 for _ in range(n_params)]
                    starting_iter_idx -= 1
                    iter_idx = starting_iter_idx
            elif idxs[iter_idx] == lens[iter_idx]-1:
                    iter_idx+=1
            else:
                idxs[iter_idx]+=1

        return self

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)

    def decision_function(self, X):
        return self.best_model.decision_function(X)

    def evaluate(self, X, y):
        return self.best_model.evaluate(X, y)
