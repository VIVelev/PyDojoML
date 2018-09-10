from ..utils import (
    BaseModel,
    MethodNotSupportedError,

    set_kernel,
    svm_problem,
    svm_parameter,
    svm_train,
    svm_predict,

    mean_squared_error,
)

__all__ = [
    "NuSVR",
]

class NuSVR(BaseModel):
    """Nu-Support Vector Machine Regressor
    
    ... (more documentation)
    
    Parameters:
    -----------
    nu : float, optional
    kernel : string, optional
    degree : integer, optional
    gamma : "auto" or float, optional
    
    """

    def __init__(self, nu=0.5, kernel="rbf", degree=3, gamma="auto"):
        super().__init__()

        self._estimator = None
        self.nu = nu
        self.kernel = set_kernel(kernel)
        self.degree = degree
        self.gamma = gamma

    def fit(self, X, y):
        X, y = super().fit(X, y)

        if self.gamma.upper() == "AUTO":
            self.gamma = 1.0/len(X[0])

        problem = svm_problem(y, X)
        parameter = svm_parameter(
            "-s 3 -n " + str(self.nu) +
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
        raise MethodNotSupportedError("Probability prediction are not supported for NuSVR model.")

    def decision_function(self, X):
        X = super().predict(X)
        *_, decision_values = svm_predict([0 for _ in X], X, self._estimator, options="-q")
        return decision_values

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Mean Squared Error: {mean_squared_error(y, self.predict(X))}"
        )
