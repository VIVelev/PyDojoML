import numpy as np
from ..base import BaseModel

__all__ = [
    "AdaBoostClassifier",
]

class AdaBoostClassifier(BaseModel):
    
    def __init__(self, base_estimator=None, n_iterations=50, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X, y = super().fit(X, y)
        data = np.column_stack((X, y))
        size = data.shape[0]

        for n_iter in range(self.n_iterations):
            idxs = np.random.choice(list(range(data.shape[0])), size)
            X, y = data[idxs, :-1], data[idxs, -1]

            self.base_estimator.fit(X, y)
            pred_y = np.array(self.base_estimator.predict(X))

            data = np.vstack((
                data,
                np.column_stack((
                    X[y != pred_y, :], y[y != pred_y]
                ))
            ))
            np.random.shuffle(data)

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def decision_function(self, X):
        return self.base_estimator.decision_function(X)

    def evaluate(self, X, y):
        self.base_estimator.evaluate(X, y)
