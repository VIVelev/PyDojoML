import numpy as np

from .impurity_measurements import (
    gini_impurity,
    entropy,
    info_gain,
) 

from .structure import (
    Question,
    Node,
    Leaf,
)

__all__ = [
    "split",
    "find_best_question",
    "pick_rand_question",
    "build_tree",
    "build_extra_tree",
    "tree_predict",
    "print_tree",
]

def split(X, Y, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """

    true_X, false_X = [], []
    true_Y, false_Y = [], []

    for x, y in zip(X, Y):
        if question.match(x):
            true_X.append(x)
            true_Y.append(y)

        else:
            false_X.append(x)
            false_Y.append(y)

    return (np.array(true_X), np.array(false_X),
            np.array(true_Y), np.array(false_Y))

def find_best_question(X, y, criterion):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
    """
    
    measure_impurity = gini_impurity if criterion == "gini" else entropy

    current_impurity = measure_impurity(y)
    best_info_gain = 0
    best_question = None

    for feature_n in range(X.shape[1]):
        for value in set(X[:, feature_n]):
            q = Question(feature_n, value)
            _, _, true_y, false_y = split(X, y, q)

            current_info_gain = info_gain(current_impurity, true_y, false_y, criterion)
            if current_info_gain >= best_info_gain:
                best_info_gain = current_info_gain
                best_question = q

    return best_info_gain, best_question

def pick_rand_question(X, y, criterion):
    rand_features = [
        np.random.randint(0, X.shape[1]) for _ in range(int(np.sqrt(X.shape[1])))
    ]
    rand_questions = [
        Question(i, np.random.choice(X[:, i])) for i in rand_features
    ]

    measure_impurity = gini_impurity if criterion == "gini" else entropy

    current_impurity = measure_impurity(y)
    best_info_gain = 0
    best_question = None

    for q in rand_questions:
        _, _, true_y, false_y = split(X, y, q)

        current_info_gain = info_gain(current_impurity, true_y, false_y, criterion)
        if current_info_gain >= best_info_gain:
            best_info_gain = current_info_gain
            best_question = q
    
    return best_info_gain, best_question

def build_tree(X, y, criterion, max_depth, current_depth=1):
    """Builds the decision tree.
    """

    # check for max_depth accomplished
    if max_depth >= 0 and current_depth >= max_depth:
        return Leaf(y)

    # check for 0 gain
    gain, question = find_best_question(X, y, criterion)
    if gain == 0:
        return Leaf(y)

    # split
    true_X, false_X, true_y, false_y = split(X, y, question)

    # Build the `true` branch of the tree recursively
    true_branch = build_tree(
        true_X, true_y,
        criterion,
        max_depth,
        current_depth=current_depth+1
    )
    
    # Build the `false` branch of the tree recursively
    false_branch = build_tree(
        false_X, false_y,
        criterion,
        max_depth,
        current_depth=current_depth+1
    )

    # returning the root of the tree/subtree
    return Node(
        question=question,
        true_branch=true_branch,
        false_branch=false_branch
    )

def build_extra_tree(X, y, criterion, max_depth, current_depth=1):
    """Builds the extremely randomized decision tree.
    """

    # check for max_depth accomplished
    if max_depth >= 0 and current_depth >= max_depth:
        return Leaf(y)

    # pick the random question
    gain, question = pick_rand_question(X, y, criterion)
    if gain == 0:
        return Leaf(y)

    # split
    true_X, false_X, true_y, false_y = split(X, y, question)

    # Build the `true` branch of the tree recursively
    true_branch = build_tree(
        true_X, true_y,
        criterion,
        max_depth,
        current_depth=current_depth+1
    )
    
    # Build the `false` branch of the tree recursively
    false_branch = build_tree(
        false_X, false_y,
        criterion,
        max_depth,
        current_depth=current_depth+1
    )

    # returning the root of the tree/subtree
    return Node(
        question=question,
        true_branch=true_branch,
        false_branch=false_branch
    )

def tree_predict(x, root, proba=False, regression=False):
    """Predicts a probabilities/value/label for the sample x.
    """

    if isinstance(root, Leaf):
        if proba:
            return root.probabilities
        elif regression:
            return root.mean
        else:
            return root.most_frequent

    if root.question.match(x):
        return tree_predict(x, root.true_branch, proba=proba, regression=regression)
    else:
        return tree_predict(x, root.false_branch, proba=proba, regression=regression)

def print_tree(root, space=' '):
    """Prints the Decision Tree in a pretty way.
    """

    if isinstance(root, Leaf):
        print(space + "Prediction: " + str(root.most_frequent))
        return

    print(space + str(root.question))

    print(space + "--> True:")
    print_tree(root.true_branch, space+'  ')

    print(space + "--> False:")
    print_tree(root.false_branch, space+'  ')
