__all__ = [
    "Question",
    "Node",
    "Leaf",
]

class Question:
    def __init__(self, feature_n, value):
        self.feature_n = feature_n
        self.value = value

    def mark(self, x):
        if type(self.value) is str:
            return x[self.feature_n] == self.value
        else:
            return x[self.feature_n] >= self.value

class Node:
    def __init__(self, question=None, true_branch=None, false_branch=None):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    def __init__(self, data):
        self.data = data
        self.class_ = max(set(data), key=list(data).count)
