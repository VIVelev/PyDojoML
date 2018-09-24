from ..base import BaseModel
from ..exceptions import MethodNotSupportedError

from .utils.functions import np, build_extra_tree, tree_predict, print_tree
from ..metrics.classification import accuracy_score

__all__ = [
    "ExtraClassificationTree",
]

class ExtraClassificationTree(BaseModel):
    """Extremely Randomized Classification Decision Tree model
    
    Parameters:
    -----------
    criterion : string, optional
    max_depth : positive integer, optional
    root : binary decision tree's root, optional
    
    """

    def __init__(self, criterion="gini", max_depth=-1, root=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = root

    def fit(self, X, y):
        X, y = super().fit(X, y)
        self.root = build_extra_tree(
            X, y,
            self.criterion,
            self.max_depth
        )
        return self

    def predict(self, X):
        X = super().predict(X)
        return np.array([tree_predict(x, self.root) for x in X])

    def predict_proba(self, X):
        X = super().predict_proba(X)
        return np.array([tree_predict(x, self.root, proba=True) for x in X])

    def decision_function(self, X):
        raise MethodNotSupportedError("Decision function is not supported for Extra Classification Tree model.")

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )

    def visualize(self):
        """Decision Tree visualization.
        """
        print_tree(self.root)
