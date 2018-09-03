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
