import numpy as np
from .model import UnsupervisedModel

__all__ = [
    "Clustering",
]


class Clustering(UnsupervisedModel):
    """Every clustering model inherits this class.
    """

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
