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
        best_info_gain = 0
        best_question = None

        for feature_n in range(X.shape[1]):
            for value in X[:, feature_n]:
                q = Question(feature_n, value)
                _, _, true_y, false_y = ClassificationTree.split(X, y, q)
                
                true_branch_impurity = gini_impurity(true_y)
                false_branch_impurity = gini_impurity(false_y)

                current_info_gain = info_gain(current_impurity, true_branch_impurity, false_branch_impurity)
                if current_info_gain > best_info_gain:
                    best_info_gain = current_info_gain
                    best_question = q

        return best_question

    @staticmethod
    def build_tree(X, y):
        current_impurity = gini_impurity(y)
        
        root = Node()
        root.question = ClassificationTree.find_best_question(X, y)

        true_X, false_X, true_y, false_y = ClassificationTree.split(X, y, root.question)

        true_branch_impurity = gini_impurity(true_y)
        false_branch_impurity = gini_impurity(false_y)

        info_gain_value = info_gain(current_impurity, true_branch_impurity, false_branch_impurity)

        if info_gain_value <= 0:
            root.true_branch = Leaf(true_y)
            root.false_branch = Leaf(false_y)

        else:
            root.true_branch = ClassificationTree.build_tree(true_X, true_y)
            root.false_branch = ClassificationTree.build_tree(false_X, false_y)

        return root

    def fit(self, X, y):
        self.root = ClassificationTree.build_tree(X, y)

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
