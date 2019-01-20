from .utils import (
    np,
    Classifier,
    Sigmoid,
    CrossEntropy,
    L2,
    accuracy_score,
)

__all__ = [
    "LogisticRegression",
]


class LogisticRegression(Classifier):
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

    def __init__(self, alpha=0.1, loss=CrossEntropy(), regularizer=L2(0.001), verbose=False):
        self.alpha = alpha
        self.loss = loss
        self.regularizer = regularizer
        self.verbose = verbose
        
        self._intercept = 0
        self._coefs = []
        self._activation_func = Sigmoid()
    
    def _backprop(self, X, y, a, z):
        dz = self.loss.gradient(y, a) * self._activation_func.gradient(z)
        dintercept = np.mean(dz)
        dcoefs = 1/y.size * X.T @ dz + 1/y.size * self.regularizer.gradient(self._coefs)

        return dintercept, dcoefs

    def _forwardprop(self, X, y):
        z = self.decision_function(X)
        a = self._activation_func(z)
        current_cost = np.mean(self.loss(y, a)) + 1/y.size * self.regularizer(self._coefs)

        return z, a, current_cost

    def fit(self, X, y):
        X, y = super().fit(X, y)
        y = y.reshape(-1, 1)

        # Init parameters
        self._intercept = dintercept = 0
        self._coefs = dcoefs = np.zeros((X.shape[1], 1), dtype=np.float32)
        
        best_cost = 1e6
        current_cost = 0
        n_iters = 1
        z, a, current_cost = self._forwardprop(X, y)
        
        while n_iters <= X.shape[0] or best_cost > current_cost:
            best_cost = current_cost
            if self.verbose and n_iters % 10 == 0:
                print("--------------------------")
                print(f"{n_iters}th iteration")
                print(f"Loss: {best_cost}")

            # Calculate the gradient
            dintercept, dcoefs = self._backprop(X, y, a, z)
            # Update
            self._intercept -= self.alpha * dintercept
            self._coefs -= self.alpha * dcoefs
            
            # Compute the current cost
            n_iters += 1
            z, a, current_cost = self._forwardprop(X, y)

        # Go back to the local minimum
        self._intercept += self.alpha * dintercept
        self._coefs += self.alpha * dcoefs
        return self

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        return self._activation_func(self.decision_function(X))

    def decision_function(self, X):
        X = super().decision_function(X)
        return X @ self._coefs + self._intercept

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))
