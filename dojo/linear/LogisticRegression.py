from .utils import (
    np,
    BaseModel,
    Sigmoid,
    CrossEntropy,
    L2,
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
    alpha : float number, learning rate also known as alpha factor, optional
    loss : Dojo-Loss, optional
    regularizer, Dojo-Regularizer, optional
    verbose : boolean, optional
    
    """

    def __init__(self, alpha=0.1, loss=CrossEntropy(), regularizer=L2(0), verbose=False):
        self.alpha = alpha
        self.loss = loss
        self.regularizer = regularizer
        self.verbose = verbose
        
        self.intercept = 0
        self.coefs = []
        self._activation_func = Sigmoid()

    def fit(self, X, y):
        X, y = super().fit(X, y)
        y = y.reshape(-1, 1)
        m, n = X.shape

        self.intercept = dintercept = 0
        self.coefs = dcoefs = np.zeros((n, 1), dtype=np.float32)
        
        best_loss = 1e6
        n_iters = 1
        l = 0

        z = self.decision_function(X)
        a = self._activation_func(z)
        l = np.mean(self.loss(y, a)) + 1/m * self.regularizer(self.coefs)
        
        while n_iters <= m or best_loss > l:
            best_loss = l
            
            # Compute the derivatives
            dz = self.loss.gradient(y, a) * self._activation_func.gradient(z)
            dintercept = np.mean(dz)
            dcoefs = 1/m * X.T @ dz + 1/m * self.regularizer.gradient(self.coefs)
            
            if self.verbose and n_iters % 10 == 0:
                print("--------------------------")
                print(f"{n_iters}th iteration")
                print(f"Loss: {best_loss}")

            self.intercept -= self.alpha * dintercept
            self.coefs -= self.alpha * dcoefs
            
            n_iters += 1
            z = self.decision_function(X)
            a = self._activation_func(z)
            l = np.mean(self.loss(y, a)) + 1/m * self.regularizer(self.coefs)

        self.intercept += self.alpha * dintercept
        self.coefs += self.alpha * dcoefs
        return self

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        return self._activation_func(self.decision_function(X))

    def decision_function(self, X):
        X = super().decision_function(X)
        return X @ self.coefs + self.intercept

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))
