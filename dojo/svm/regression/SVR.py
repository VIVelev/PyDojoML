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
    "SVR",
]

class SVR(BaseModel):
    """Epsilon-Support Vector Machine Regressor
    
    Parameters:
    -----------
    epsilon : float, optional
    kernel : string, optional
    degree : integer, optional
    gamma : "auto" or float, optional
    verbose : boolean, optional
    
    """

    def __init__(self, epsilon=0.01, kernel="rbf", degree=3, gamma="auto", verbose=False):
        self._estimator = None
        self.epsilon = epsilon
        self.kernel = set_kernel(kernel)
        self.degree = degree
        self.gamma = gamma
        self.verbose = verbose

    def fit(self, X, y):
        X, y = super().fit(X, y)

        if self.gamma.upper() == "AUTO":
            self.gamma = 1.0/X.shape[0]

        problem = svm_problem(y, X)
        param_str = f"""
            -s 3
            -p {self.epsilon}
            -t {self.kernel}
            -d {self.degree}
            -g {self.gamma}
        """
        if not self.verbose:
            param_str += " -q"
        parameter = svm_parameter(param_str)

        self._estimator = svm_train(problem, parameter)
        return self

    def predict(self, X):
        X = super().predict(X)
        predictions, *_ = svm_predict([0 for _ in X], X, self._estimator, options="-q")
        return predictions

    def predict_proba(self, X):
        raise MethodNotSupportedError("Probability prediction are not supported for Support Vector Machine.")

    def decision_function(self, X):
        X = super().decision_function(X)
        *_, decision_values = svm_predict([0 for _ in X], X, self._estimator, options="-q")
        return decision_values

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Mean Squared Error: {mean_squared_error(y, self.predict(X))}"
        )
