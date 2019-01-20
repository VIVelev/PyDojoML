import numpy as np

from ..base import Classifier
from ..exceptions import MethodNotSupportedError
from ..metrics.classification import accuracy_score


class NaiveBayes(Classifier):
    """Naive Bayes Classifier
    
    P(A|B) = P(B|A) * P(A) / P(B)
    
    Parameters:
    -----------
    eps : float, laplacian smoothing coefficient, optional
    
    """

    def __init__(self, eps=1e-18):
        self.eps = eps

        self._X = []
        self._y = []
        self._labels = []

    def fit(self, X, y):
        self._X, self._y = super().fit(X, y)
        self._labels = np.unique(self._y)
        return self

    def predict(self, X):
        return np.array([
            self._labels[np.argmax(proba)] for proba in self.predict_proba(X)
        ])

    def predict_proba(self, X):
        X = super().predict_proba(X)
        res = np.array([
            [
                self._calc_prior(label) * np.product(
                    [self._calc_likelihood(x, i, label) / self._calc_evidence(x, i) for i in range(len(x))]
                ) for label in self._labels
            ] for x in X
        ])

        return res / (np.sum(res, axis=1, keepdims=True) + self.eps)

    def _calc_likelihood(self, x, i, label):
        tmp = self._X[self._y == label, :]
        return np.count_nonzero(tmp[:, i] == x[i]) / tmp.shape[0]

    def _calc_prior(self, label):
        return np.count_nonzero(self._y == label) / self._y.size

    def _calc_evidence(self, x, i):
        val = np.count_nonzero(self._X[:, i] == x[i]) / self._X.shape[0]
        if val == 0:
            return self.eps
        else:
            return val

    def decision_function(self, X):
        raise MethodNotSupportedError("Decision function is not supported for Naive Bayes Classifier.")

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))
