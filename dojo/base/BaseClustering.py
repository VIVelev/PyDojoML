import numpy as np

__all__ = [
    "BaseClustering",
]

class BaseClustering:
    """Every Dojo-Clustering model inherits this class.
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
        """Returns the specified parameters for the clustering.

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
        clustering.set_params(n_clusters=5)
        ```
        
        Returns:
        --------
        void : void, returns nothing
        
        """

        for k, v in params.items():
            vars(self)[k] = v

    def cluster(self, X):
        """Clustering - assigns clusters to the samples
        
        Parameters:
        -----------
        X : array-like, shape (m, n), the samples
        
        Returns:
        --------
        clusters : array-like, shape (m, ), the assigned clusters
        
        """

        return np.array(X, dtype=np.float32)
