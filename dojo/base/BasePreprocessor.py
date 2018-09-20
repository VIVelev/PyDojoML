import numpy as np

__all__ = [
    "BasePreprocessor",
]

class BasePreprocessor:
    """Every Dojo-Preprocessor inherits this class.
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
        """Returns the specified parameters for the current preprocessor.

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
        preproc.set_params(C=0.34, kernel="rbf")
        ```
        
        Returns:
        --------
        void : void, returns nothing
        
        """

        for k, v in params.items():
            vars(self)[k] = v

    def fit(self, X, y=None):
        """Passing the data to transform to the data-preprocessor.
        
        Parameters:
        -----------
        X : array-like data
        y : the labels, optional
        
        """

        if y is None:
            return np.array(X, dtype=np.float32)
        else:
            X, y = (np.array(X, dtype=np.float32), 
                np.array(y, dtype=np.float32))
            assert X.shape[0] == y.shape[0]
            return X, y

    def transform(self, X):
        """Transforms the data given.
        
        Parameters:
        -----------
        X : array-like data        
        
        """

        return np.array(X, dtype=np.float32)

    def fit_transform(self, X, y=None):
        """Fit-Transform shortcut function.
        
        Parameters:
        -----------
        X : array-like data
        y : the labels, optional
        
        """

        if y is None:
            self.fit(X)
        else:
            self.fit(X, y)
        return self.transform(X)
