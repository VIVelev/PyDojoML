import numpy as np
from ..base import BaseModel

def prop(x, s):
    return np.count_nonzero(s == x)/s.size

def gini_impurity(s):
    return 1 - sum(prop(s[i], s)**2 for i in range(s.size))

def entropy(s):
    return -sum(
        p*np.log(p) for i in range(s.size) for p in [prop(s[i], s)]
    )

def info_gain(current_impurity, impurity_after_action):
    return current_impurity - impurity_after_action

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
        self.class_ = max(set(data), key=data.count)
