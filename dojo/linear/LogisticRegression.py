from .utils import (
    np,
    BaseModel,
    sigmoid,
    accuracy_score,
)

__all__ = [
    "LogisticRegression",
]

class LogisticRegression(BaseModel):
    """Logistic Regression classification model.
    
    In statistics, the logistic model (or logit model) is a statistical model
    that is usually taken to apply to a binary dependent variable. In regression
    analysis, logistic regression or logit regression is estimating the
    parameters of a logistic model.
    
    Parameters:
    -----------
    intercept : float number, optional
    coefs : list of float numbers, shape (n_features,), optional
    C : float number, weight given to the loss
    compared to the regularization, optional
    lr : float number, learning rate also known as alpha factor, optional
    verbose : boolean, optional
    
    """

    def __init__(self, intercept=0, coefs=[], C=1.0, lr=0.01, verbose=False):
        self.intercept = intercept
        self.coefs = coefs
        self.C = C
        self.lr = lr
        self.verbose = verbose

        self._X, self._y = [], []

    def _loss(self):
        y_pred = np.array([1-(1e-12) if self.predict([x])[0] == 1 else 1e-12 for x in self._X])
        m, _ = self._X.shape

        return -1/m * np.sum(
            self._y * np.log(y_pred) + (np.ones(m)-self._y) * np.log(np.ones(m)-y_pred)
        ) + 1/self.C * 1/2*m * np.sum(self.coefs**2)

    def _gradient(self):
        y_pred = np.array([self.predict([x])[0] for x in self._X])
        m, n = self._X.shape

        grad = np.zeros(n+1, dtype=np.float32)
        grad[0] = 1/m * np.sum(y_pred - self._y)
        grad[1:] = (self._X.T @ (y_pred - self._y)).T * 1/m + 1/self.C * 1/m * self.coefs
        
        return grad

    def fit(self, X, y):
        self._X, self._y = super().fit(X, y)
        m, n = self._X.shape

        self.intercept = 0
        self.coefs = np.zeros(n, dtype=np.float32)
        
        best_loss = 1e6
        grad = None
        n_iters = 1
        l = self._loss()

        while n_iters < m or best_loss > l:
            best_loss = l
            grad = self._gradient()
            
            if self.verbose and n_iters % 10 == 0:
                print("--------------------------")
                print(f"{n_iters}th iteration")
                print(f"Loss: {best_loss}")

            self.intercept -= self.lr * grad[0]
            self.coefs -= self.lr * grad[1:]

            n_iters += 1
            l = self._loss()

        self.intercept += self.lr * grad[0]
        self.coefs += self.lr * grad[1:]
        return self

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def decision_function(self, X):
        X = super().decision_function(X)
        return (X @ self.coefs).reshape(1, -1) + np.array([self.intercept for _ in range(X.shape[0])])

    def evaluate(self, X, y):
        X, y = super().evaluate(X, y)
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )
