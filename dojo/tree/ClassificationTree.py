from ..base import BaseModel

from .utils.impurity_measurements import gini_impurity, entropy
from .utils.functions import build_tree, tree_predict

__all__ = [
    "ClassificationTree",
]

class ClassificationTree(BaseModel):
    """Classification Decision Tree Model
    
    ... (more documentation)
    
    Parameters:
    -----------
    criterion : string, optional
    
    """

    def __init__(self, criterion="gini"):
        self.criterion = criterion
        self.impurity_func = gini_impurity if self.criterion == "gini" else entropy

        self.root = None

    def fit(self, X, y):
        X, y = super().fit(X, y)
        self.root = build_tree(X, y, self.impurity_func)

    def predict(self, X):
        X = super().predict(X)
        return [tree_predict(x, root) for x in X for root in [self.root]]

    def predict_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        pass
