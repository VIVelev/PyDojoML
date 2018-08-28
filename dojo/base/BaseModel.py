import numpy as np

__all__ = [
    "BaseModel",
]

class BaseModel:
    """Every Dojo-Model inherits this class.
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
        """Returns the specified parameters for the current model.

        Parameters:
        -----------
        keys : variable sized list, containing the names of the requested parameters

        Returns:
        --------
        values : list or dictionary, if any `keys` are specified
        those named parameters' values are returned, otherwise
        all parameters are returned as a dictionary
        
        """

        if len(keys) == 0:
            return vars(self)
        else:
            return [vars(self)[k] for k in keys]

    def set_params(self, **params):
        """Sets new values to the specified parameters.
        
        Parameters:
        -----------
        params : variable sized dictionary, n key-word arguments
        Example:
        ```
        model.set_params(C=0.34, kernel="rbf")
        ```
        
        Returns:
        --------
        void : void, returns nothing
        
        """

        for k, v in params.items():
            vars(self)[k] = v

    def fit(self, X, y):
        """Fits the given model to the data and labels provided.
        
        Parameters:
        -----------
        X : matrix, shape (n_samples, n_features)
        The samples, the train data.

        y : vector, shape (n_samples,)
        The target labels.
        
        Returns:
        --------
        self : instance of the model itself (`self`)
        
        """

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        assert X.shape[0] == y.shape[0]

        return X, y

    def predict(self, X):
        """Predicts the labels of the given data.
        
        Parameters:
        -----------
        X : matrix, shape (n_samples, n_features)
        The samples.
        
        Returns:
        --------
        y : vector, shape (n_samples,)
        The predicted labels.
        
        """

        return np.array(X, dtype=np.float32)

    def predict_proba(self, X):
        """Probability prediction of the given data.
        
        Parameters:
        -----------
        X : matrix, shape (n_samples, n_features)
        The samples.
        
        Returns:
        --------
        y : vector, shape (n_samples,)
        The predicted probabilities.
        
        """

        return np.array(X, dtype=np.float32)
    
    def decision_function(self, X):
        """Applies only the hypothesis (decision function)
        to the given data.
        
        Parameters:
        -----------
        X : matrix, shape (n_samples, n_features)
        The samples.
        
        Returns:
        --------
        y : vector, shape (n_samples,)
        The non shrank, raw values acquired from the outputs of
        the hypothesis. No shrinking, probability functions is applied.
        
        """

        return np.array(X, dtype=np.float32)

    def fit_predict(self, X, y, X_):
        """Shortcut to `model.fit(X, y); return model.predict(X_)`.
        
        Parameters:
        -----------
        X : matrix, shape (n_samples, n_features)
        The samples, the train data.

        y : vector, shape (n_samples,)
        The target labels.

        X_ : matrix, shape (m_samples, m_features)
        The samples which labels to predict.
        
        Returns:
        --------
        y : vector, shape (m_samples,)
        The predicted labels.
        
        """

        self.fit(X, y)
        return self.predict(X_)

    def evaluate(self, X, y):
        """Error/Accuracy examination of the model.
        
        Parameters:
        -----------
        X : matrix, shape (n_samples, n_features)
        The samples.

        y : vector, shape (n_samples,)
        The target labels.
        
        Returns:
        --------
        void : nothing, prints a summary of the examination
        
        """

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        assert X.shape[0] == y.shape[0]

        return X, y
