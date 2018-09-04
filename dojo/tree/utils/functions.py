import numpy as np

from .impurity_measurements import info_gain

from .structure import (
    Question,
    Node,
    Leaf
)

__all__ = [
    "split",
    "find_best_question",
    "build_tree",
    "tree_predict",
]

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

def find_best_question(X, y, impurity_func):
    current_impurity = impurity_func(y)
    best_info_gain = 0
    best_question = None

    for feature_n in range(X.shape[1]):
        for value in set(X[:, feature_n]):
            q = Question(feature_n, value)
            _, _, true_y, false_y = split(X, y, q)

            current_info_gain = info_gain(current_impurity, true_y, false_y, impurity_func)
            if current_info_gain >= best_info_gain:
                best_info_gain = current_info_gain
                best_question = q

    return best_info_gain, best_question

def build_tree(X, y, impurity_func):
    gain, question = find_best_question(X, y, impurity_func)
    if gain == 0:
        return Leaf(y)

    true_X, false_X, true_y, false_y = split(X, y, question)

    true_branch = build_tree(true_X, true_y, impurity_func)
    false_branch = build_tree(false_X, false_y, impurity_func)

    return Node(
        question=question,
        true_branch=true_branch,
        false_branch=false_branch
    )

def tree_predict(x, root):
    if isinstance(root, Leaf):
        return root.class_

    if root.question.mark(x):
        return tree_predict(x, root.true_branch)
    else:
        return tree_predict(x, root.false_branch)
