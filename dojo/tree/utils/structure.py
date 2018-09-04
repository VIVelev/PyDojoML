import numpy as np

__all__ = [
    "Question",
    "Node",
    "Leaf",
]

class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    @staticmethod
    def is_numeric(value):
        """Test if a value is numeric.
        """
        return (
            isinstance(value, int) or
            isinstance(value, float) or

            isinstance(value, np.int8) or
            isinstance(value, np.int16) or
            isinstance(value, np.int32) or
            isinstance(value, np.int64) or

            isinstance(value, np.float16) or
            isinstance(value, np.float32) or
            isinstance(value, np.float64) or
            isinstance(value, np.float128)
        )

    def __init__(self, feature_n, value):
        self.feature_n = feature_n
        self.value = value

    def __repr__(self):
        if Question.is_numeric(self.value):
            return 'Is feature[' + str(self.feature_n) + '] >= ' + str(self.value) + '?'
        else:
            return 'Is feature[' + str(self.feature_n) + '] == ' + str(self.value) + '?'

    def __str__(self):
        return self.__repr__()

    def match(self, x):
        if Question.is_numeric(self.value):
            return x[self.feature_n] >= self.value
        else:
            return x[self.feature_n] == self.value

class Node:
    """A Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self, question=None, true_branch=None, false_branch=None):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    """A Leaf node classifies data.
    """

    def __init__(self, data):
        self.data = data
        self.probabilities = dict(
            (data[i], list(data).count(data[i])/len(data)) for i in range(len(data))
        )
        self.most_frequent = max(set(data), key=list(data).count)
        self.mean = np.mean(data)
