import numpy as np

__all__ = [
    "BaseModel",
]

class BaseModel:
    """Every ML Model inherits this class.
    """

    def __init__(self):
        pass

    def __repr__(self):
        res = str(self.__class__)
        res = res.split(".")[-1][:-2]
        res += "("

        for k, v in vars(self).items():
            if k[0] != '_':
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
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        assert X.shape[0] == y.shape[0]

        return X, y

    def predict(self, X):
        return np.array(X, dtype=np.float32)

    def predict_proba(self, X):
        return np.array(X, dtype=np.float32)
    
    def decision_function(self, X):
        return np.array(X, dtype=np.float32)

    def fit_predict(self, X, y, X_test):
        self.fit(X, y)
        return self.predict(X_test)

    def evaluate(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        assert X.shape[0] == y.shape[0]

        return X, y
