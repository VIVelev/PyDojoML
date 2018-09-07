from .utils import (
    np,
    linalg,

    BaseModel,
    MethodNotSupportedError,

    mean_squared_error,
)

__all__ = [
    "Lasso",
]

class Lasso(BaseModel):
    """L1 regularized Linear Regression model.
    
    In statistics and machine learning, lasso (least absolute shrinkage and selection operator)
    (also Lasso or LASSO) is a regression analysis method that performs both variable selection
    and regularization in order to enhance the prediction accuracy and
    interpretability of the statistical model it produces.
    
    Parameters:
    -----------
    intercept : float number, optional
    coefs : list of float numbers, shape (n_features,), optional
    alpha : float number, regularization weight (multiplies L1 term), optional
    verbose : boolean, optional

    """

    def __init__(self, intercept=0, coefs=[], alpha=1.0, verbose=False):
        self.intercept = intercept
        self.coefs = coefs
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, X, y):
        X, y = super().fit(X, y)

        m, n = X.shape
        X = np.hstack((
            np.array([[1.0] for _ in range(m)]),
            X
        ))

        L = np.eye(n, n)
        L[0, 0] = 0

        if self.verbose:
            print("-----------------------------------------")
            print("Fitting...")
        self.intercept, *self.coefs = linalg.inv(X.T @ X + self.alpha * L) @ X.T @ y
        if self.verbose:
            print("The model has been fitted successfully!")
            print("-----------------------------------------")

        return self

    def predict(self, X):
        X = super().predict(X)
        return [self.intercept + np.array(self.coefs).T @ x for x in X]
    
    def predict_proba(self, X):
        raise MethodNotSupportedError("Probability predictions are not supported for LASSO Regression.")
    
    def decision_function(self, X):
        raise MethodNotSupportedError("Use `predict` method instead.")

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Mean Squared Error: {mean_squared_error(y, self.predict(X))}"
        )
