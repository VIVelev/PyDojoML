from abc import ABC, abstractmethod

__all__ = [
    "BaseModel"
]

class BaseModel:
    """Every ML Model inherits this class
    """

    def __init__(self, **params):
        self._params = params

    def __repr__(self):
        res = str(self.__class__)
        res = res.split(".")[-1][:-2]
        res += "("

        for k, v in self._params.items():
            if type(v) is str:
                res += f"\n    {k}='{v}',"
            else:
                res += f"\n    {k}={v},"

        return res+"\n)"
    
    def __str__(self):
        return self.__repr__()

    def get_params(self):
        return self._params

    def set_params(self, **params):
        for k, v in params.items():
            self._params[k] = v

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass
    
    @abstractmethod
    def decision_function(self, X):
        pass

    def fit_predict(self, X, y, X_test):
        self.fit(X, y)
        return self.predict(X_test)

    @abstractmethod
    def evaluate(self, X, y):
        pass
