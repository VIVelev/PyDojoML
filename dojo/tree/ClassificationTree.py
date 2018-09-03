from .utils import (
    np,

    BaseModel,

    gini_impurity,
    entropy,
    info_gain,

    Question,
    Node,
    Leaf,
)

__all__ = [
    "ClassificationTree",
]

class ClassificationTree(BaseModel):
    def __init__(self, metric="gini"):
        self.metric = metric
        self.root = None
    
    @staticmethod
    def split(X, Y, question):
        true_X, false_X = [], []
        true_Y, false_Y = [], []

        for x, y in X, Y:
            if question.mark(x):
                true_X.append(x)
                true_Y.append(y)

            else:
                false_X.append(x)
                false_Y.append(y)

        return true_X, false_X, true_Y, false_Y

    @staticmethod
    def find_best_question(X, y):
        current_impurity = gini_impurity(y)

        for feature_n in range(X.shape[1]):
            for value in X[:, feature_n]:
                q = Question(feature_n, value)

    def fit(self, X, y):
        pass

    @staticmethod
    def predict_one(x, root):
        if type(root) is Leaf:
            return root.class_

        if root.question.mark(x):
            ClassificationTree.predict_one(x, root.true_branch)
        else:
            ClassificationTree.predict_one(x, root.false_branch)

        return None

    def predict(self, X):
        return [ClassificationTree.predict_one(x, root) for x in X for root in [self.root]]

    def predict_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        pass
