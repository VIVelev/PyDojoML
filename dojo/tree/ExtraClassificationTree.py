from ..base import BaseModel
from ..exceptions import MethodNotSupportedError

from .utils.functions import build_extra_tree, tree_predict, print_tree
from ..metrics.classification import accuracy_score

__all__ = [
    "ExtraClassificationTree",
]

class ExtraClassificationTree(BaseModel):
    """Extremely Randomized Classification Decision Tree model
    
    ... (more documentation)
    
    Parameters:
    -----------
    criterion : string, optional
    max_depth : positive integer, optional
    
    """

    def __init__(self, criterion="gini", max_depth=-1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X, y = super().fit(X, y)
        self.root = build_extra_tree(
            X, y,
            self.criterion,
            self.max_depth
        )

    def predict(self, X):
        X = super().predict(X)
        return [tree_predict(x, root) for x in X for root in [self.root]]

    def predict_proba(self, X):
        X = super().predict(X)
        return [tree_predict(x, root, proba=True) for x in X for root in [self.root]]

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
