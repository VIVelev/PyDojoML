from .utils import (
    BaseModel,

    set_kernel,
    svm_problem,
    svm_parameter,
    svm_train,
    svm_predict,

    accuracy_score,
)

__all__ = [
    "SVC",
]

class SVC(BaseModel):
    """C-Support Vector Machine Classifier
    
    ... (more documentation)
    
    Parameters:
    -----------
    C : float, optional
    kernel : string, optional
    degree : integer, optional
    gamma : "auto" or float, optional
    
    """

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="auto"):
        super().__init__()

        self._estimator = None
        self.C = C
        self.kernel = set_kernel(kernel)
        self.degree = degree
        self.gamma = gamma

    def fit(self, X, y):
        X, y = super().fit(X, y)

        if self.gamma.upper() == "AUTO":
            self.gamma = 1.0/X.shape[0]

        problem = svm_problem(y, X)
        parameter = svm_parameter(
            "-s 0 -c " + str(self.C) +
            " -t " + str(self.kernel) +
            " -d " + str(self.degree) +
            " -g " + str(self.gamma)
        )
        self._estimator = svm_train(problem, parameter)
        return self

    def predict(self, X):
        X = super().predict(X)
        predictions, *_ = svm_predict([0 for _ in X], X, self._estimator, options="-q")
        return predictions

    def predict_proba(self, X):
        X = super().predict(X)
        *_, probabilities = svm_predict([0 for _ in X], X, self._estimator, options="-q -b 1")
        return probabilities

    def decision_function(self, X):
        X = super().predict(X)
        *_, decision_values = svm_predict([0 for _ in X], X, self._estimator, options="-q")
        return decision_values

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )
