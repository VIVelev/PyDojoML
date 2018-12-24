import numpy as np
from ..base import BasePreprocessor

__all__ = [
    "OneHotEncoder",
]


class OneHotEncoder(BasePreprocessor):
    # TODO: extend the functionality to a matrix handler

    def __init__(self):
        self.n_unique_values = 0

    def fit(self, X):
        self.n_unique_values = np.unique(X).size
        return self

    def transform(self, X):
        encoded = []
        for i in range(X.size):
            vec = np.zeros(self.n_unique_values)
            vec[int(X[i])] = 1
            encoded.append(vec)
        
        return np.array(encoded)
