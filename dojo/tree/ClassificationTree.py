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
    def __init__(self, criterion="gini"):
        self.criterion = criterion
        self.impurity_func = gini_impurity if self.criterion == "gini" else entropy

        self.root = None
    
    @staticmethod
    def split(X, Y, question):
        true_X, false_X = [], []
        true_Y, false_Y = [], []

        for x, y in zip(X, Y):
            if question.mark(x):
                true_X.append(x)
                true_Y.append(y)

            else:
                false_X.append(x)
                false_Y.append(y)

        return (np.array(true_X), np.array(false_X),
                np.array(true_Y), np.array(false_Y))

    @staticmethod
    def find_best_question(X, y, impurity_func):
        current_impurity = impurity_func(y)
        best_info_gain = 0
        best_question = None

        for feature_n in range(X.shape[1]):
            for value in set(X[:, feature_n]):
                q = Question(feature_n, value)
                _, _, true_y, false_y = ClassificationTree.split(X, y, q)

                current_info_gain = info_gain(current_impurity, true_y, false_y, impurity_func)
                if current_info_gain >= best_info_gain:
                    best_info_gain = current_info_gain
                    best_question = q

        return best_info_gain, best_question

    @staticmethod
    def build_tree(X, y, impurity_func):
        gain, question = ClassificationTree.find_best_question(X, y, impurity_func)
        if gain == 0:
            return Leaf(y)

        true_X, false_X, true_y, false_y = ClassificationTree.split(X, y, question)

        true_branch = ClassificationTree.build_tree(true_X, true_y, impurity_func)
        false_branch = ClassificationTree.build_tree(false_X, false_y, impurity_func)

        return Node(
            question=question,
            true_branch=true_branch,
            false_branch=false_branch
        )

    def fit(self, X, y):
        X, y = super().fit(X, y)

        self.root = ClassificationTree.build_tree(X, y, self.impurity_func)

    @staticmethod
    def predict_one(x, root):
        if isinstance(root, Leaf):
            return root.class_

        if root.question.mark(x):
            return ClassificationTree.predict_one(x, root.true_branch)
        else:
            return ClassificationTree.predict_one(x, root.false_branch)

    def predict(self, X):
        X = super().predict(X)
        return [ClassificationTree.predict_one(x, root) for x in X for root in [self.root]]

    def predict_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        pass
