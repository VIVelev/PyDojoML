from ..base import BaseModel
from ..exceptions import MethodNotSupportedError

from .utils.functions import build_tree, tree_predict, print_tree
from ..metrics.regression import mean_squared_error

__all__ = [
    "RegressionTree",
]

class RegressionTree(BaseModel):
    """Regression Decision Tree model
    
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
        return [tree_predict(x, root, regression=True) for x in X for root in [self.root]]

    def predict_proba(self, X):
        raise MethodNotSupportedError("Probability predictions are not supported for Regression Tree model.")

    def decision_function(self, X):
        raise MethodNotSupportedError("Decision function is not supported for Regression Tree model.")

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Mean Squared Error: {mean_squared_error(y, self.predict(X))}"
        )

    def visualize(self):
        """Decision Tree visualization.
        """
        print_tree(self.root)
