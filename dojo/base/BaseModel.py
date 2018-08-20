import numpy as np

__all__ = [
    "BaseModel",
]

class BaseModel:
    """Every ML Model inherits this class
    """

    def __init__(self):
        pass

    def __repr__(self):
        res = str(self.__class__)
        res = res.split(".")[-1][:-2]
        res += "("

        for k, v in vars(self).items():
            if type(v) is str:
                res += f"\n    {k}='{v}',"
            else:
                res += f"\n    {k}={v},"

        return res+"\n)"
    
    def __str__(self):
        return self.__repr__()

    def get_params(self, *keys):
        if len(keys) == 0:
            return vars(self)
        else:
            return [vars(self)[k] for k in keys]

    def set_params(self, **params):
        for k, v in params.items():
            vars(self)[k] = v

    def fit(self, X, y):
        if type(X) is not np.ndarray:
            X = np.array(X)
        
        if type(y) is not np.ndarray:
            y = np.array(y)

        assert X.shape[0] == y.shape[0]

    def predict(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)

    def predict_proba(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
    
    def decision_function(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)

    def fit_predict(self, X, y, X_test):
        self.fit(X, y)
        return self.predict(X_test)

    def evaluate(self, X, y):
        if type(X) is not np.ndarray:
            X = np.array(X)
        
        if type(y) is not np.ndarray:
            y = np.array(y)

        assert X.shape[0] == y.shape[0]
