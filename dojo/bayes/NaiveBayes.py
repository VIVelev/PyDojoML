import numpy as np

from ..base import BaseModel
from ..exceptions import MethodNotSupportedError
from ..metrics.classification import accuracy_score


class NaiveBayes(BaseModel):
    """Naive Bayes Classifier
    
    P(A|B) = P(B|A) * P(A) / P(B)
    
    Parameters:
    -----------
    alpha : float, laplacian smoothing coefficient, optional
    
    """

    def __init__(self, alpha=1e-9):
        self.alpha = alpha

        self._X = []
        self._y = []

    def fit(self, X, y):
        self._X, self._y = super().fit(X, y)
        return self

    def predict(self, X):
        return np.array([
            np.unique(self._y)[np.argmax(prob)] for prob in self.predict_proba(X)
        ])

    def predict_proba(self, X):
        X = super().predict_proba(X)
        return np.array([
            [self.p(i, x) for i in np.unique(self._y)] for x in X
        ])

    def p(self, label, x):
        tmp = self._X[self._y == label, :]
        likelihood = np.count_nonzero(tmp == x) / tmp.shape[0]

        prior1 = np.count_nonzero(self._y == label) / self._y.size
        prior2 = np.count_nonzero(self._X == x) / self._X.shape[0] + self.alpha

        return likelihood * prior1/prior2

    def decision_function(self, X):
        raise MethodNotSupportedError("Decision function is not supported for Naive Bayes Classifier.")

    def evaluate(self, X, y):
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )
