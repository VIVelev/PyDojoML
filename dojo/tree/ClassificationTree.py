from ..base import BaseModel
from ..exceptions import MethodNotSupportedError

from .utils.functions import build_tree, tree_predict, print_tree

__all__ = [
    "ClassificationTree",
]

class ClassificationTree(BaseModel):
    """Classification Decision Tree model
    
    ... (more documentation)
    
    Parameters:
    -----------
    criterion : string, optional
    
    """

    def __init__(self, criterion="gini"):
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        X, y = super().fit(X, y)
        self.root = build_tree(
            X, y,
            self.criterion
        )

    def predict(self, X):
        X = super().predict(X)
        return [tree_predict(x, root) for x in X for root in [self.root]]

    def predict_proba(self, X):
        pass

    def decision_function(self, X):
        raise MethodNotSupportedError("Decision function is not supported for Classification Tree model.")

    def evaluate(self, X, y):
        pass

    def visualize(self):
        print_tree(self.root)
