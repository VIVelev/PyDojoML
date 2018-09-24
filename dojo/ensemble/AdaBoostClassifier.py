import numpy as np
from copy import deepcopy

from ..base import BaseModel
from ..metrics.classification import accuracy_score

__all__ = [
    "AdaBoostClassifier",
]

class AdaBoostClassifier(BaseModel):
    """Adaptive Boost Classifier - ensemble method
    
    Parameters:
    -----------
    base_classifier : dojo-classifier
    n_classifiers : integer, optional
    verbose : boolean, optional
    
    """
    
    def __init__(self, base_classifier=None, n_classifiers=50, verbose=False):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.verbose = verbose

        self._classifiers = [deepcopy(self.base_classifier) for _ in range(self.n_classifiers)]

    def fit(self, X, y):
        X, y = super().fit(X, y)

        data = np.column_stack((X, y))
        size = data.shape[0]
        idxs = list(range(size))

        for i in range(self.n_classifiers):
            if self.verbose:
                print("----------------------------------")
                print("Iteration", i+1)
                print("Training classifier...")

            X, y = data[idxs, :-1], data[idxs, -1]
            self._classifiers[i].fit(X, y)

            if self.verbose:
                print("Classifier", i+1, "trained.")
                print("Weighting missclassified samples...")

            pred_y = np.array(self._classifiers[i].predict(X))
            data = np.vstack((
                data,
                np.column_stack((
                    X[y != pred_y, :], y[y != pred_y]
                ))
            ))

            if self.verbose:
                print("Shuffling the data...")

            np.random.shuffle(data)
            idxs = np.random.choice(list(range(data.shape[0])), size)
            
        if self.verbose:
            print("=================================================================")
            print("The training session has successfully finished.")
        return self

    def predict(self, X):
        X = super().predict(X)

        predictions = []
        for x in X:
            current_predictions = []
            for classifier in self._classifiers:
                current_predictions.append(classifier.predict([x])[0])

            predictions.append(
                max(set(current_predictions), key=current_predictions.count)
            )

        return np.array(predictions)

    def predict_proba(self, X):
        X = super().predict_proba(X)
        return self._classifiers[-1].predict_proba(X)

    def decision_function(self, X):
        X = super().decision_function(X)
        return self._classifiers[-1].decision_function(X)

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )
