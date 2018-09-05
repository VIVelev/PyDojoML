import numpy as np

__all__ = [
    "prop",
    "gini_impurity",
    "entropy",
    "info_gain",
]

def prop(x, s):
    """Returns the proportion of `x` in `s`.
    """
    return list(s).count(x)/len(s)

def gini_impurity(s):
    """Calculate the Gini Impurity for a list of samples.

    See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    return 1 - sum(prop(s[i], s)**2 for i in range(len(s)))

def entropy(s):
    """Calculate the Entropy Impurity for a list of samples.
    """
    return -sum(
        p*np.log(p) for i in range(len(s)) for p in [prop(s[i], s)]
    )

def info_gain(current_impurity, true_branch, false_branch, criterion):
    """Information Gain.
    
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """

    measure_impurity = gini_impurity if criterion == "gini" else entropy
    p = float(len(true_branch)) / (len(true_branch) + len(false_branch))
    
    return current_impurity - p * measure_impurity(true_branch) - (1 - p) * measure_impurity(false_branch)
