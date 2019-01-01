import numpy as np
from ..base import BasePreprocessor

__all__ = [
    "Normalizer",
]


class Normalizer(BasePreprocessor):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std
