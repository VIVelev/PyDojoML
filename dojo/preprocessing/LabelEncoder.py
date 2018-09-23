import numpy as np
from ..base import BasePreprocessor

__all__ = [
    "LabelEncoder",
]

class LabelEncoder(BasePreprocessor):
    """Label Encoder - encodes textual data to 0,1,...
    """

    def __init__(self):
        self.n_labels = 0
        self.mapper = {}

    def fit(self, X):
        unique_X = np.unique(X)
        self.n_labels = unique_X.size
        self.mapper = {unique_X[i]:i for i in range(self.n_labels)}
        return self

    def transform(self, X):
        for i in range(len(X)):
            X[i] = self.mapper[X[i]]

        return X
