from .utils import (
    BaseModel,

    svm_problem,
    svm_parameter,
    svm_train,
    svm_predict,
)

__all__ = [
    "SVC",
]

class SVC(BaseModel):

    def __init__(self, C=1.0, kernel="rbf", degree=3.0, gamma="auto"):
        super().__init__()

        self._estimator = None
        self.C = C
        if kernel.upper() == "LINEAR":
            self.kernel = 0
        elif kernel.upper() == "POLY":
            self.kernel = 1
        elif kernel.upper() == "SIGMOID":
            self.kernel = 3
        else:
            self.kernel = 2 # RBF kernel
        self.degree = degree
        self.gamma = gamma

    def fit(self, X, y):
        X, y = super().fit(X, y)

        if self.gamma.upper() == "AUTO":
            self.gamma = 1.0/X.shape[0]

        prob = svm_problem(y, X)
        param = svm_parameter(
            "-s 0 -c " + str(self.C) +
            " -t " + str(self.kernel) +
            " -d " + str(self.degree) +
            " -g " + str(self.gamma)
        )
        self._estimator = svm_train(prob, param)
        return self

    def predict(self, X):
        X = super().predict(X)

        predictions, *_ = svm_predict([0 for _ in X], X, self._estimator)
        return predictions

    def predict_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        pass
