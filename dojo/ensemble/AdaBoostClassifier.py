import numpy as np
from copy import deepcopy

from ..base import BaseModel
from ..metrics.classification import accuracy_score

__all__ = [
    "AdaBoostClassifier",
]

class AdaBoostClassifier(BaseModel):
    
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self._estimators = [deepcopy(self.base_estimator) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        X, y = super().fit(X, y)

        data = np.column_stack((X, y))
        size = data.shape[0]
        idxs = list(range(size))

        for i in range(self.n_estimators):
            X, y = data[idxs, :-1], data[idxs, -1]

            self._estimators[i].fit(X, y)
            pred_y = np.array(self._estimators[i].predict(X))

            data = np.vstack((
                data,
                np.column_stack((
                    X[y != pred_y, :], y[y != pred_y]
                ))
            ))
            np.random.shuffle(data)
            idxs = np.random.choice(list(range(data.shape[0])), size)

        return self

    def predict(self, X):
        X = super().predict(X)

        predictions = []
        for x in X:
            current_predictions = []
            for estimator in self._estimators:
                current_predictions.append(estimator.predict([x])[0])

            predictions.append(
                max(set(current_predictions), key=current_predictions.count)
            )

        return predictions

    def predict_proba(self, X):
        X = super().predict_proba(X)
        return self._estimators[-1].predict_proba(X)

    def decision_function(self, X):
        X = super().decision_function(X)
        return self._estimators[-1].decision_function(X)

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )
