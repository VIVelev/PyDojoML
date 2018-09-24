from .utils import (
    np, linalg,
    BasePreprocessor,

    get_covariance_matrix,
)

__all__ = [
    "PrincipalComponentAnalysis",
]

class PrincipalComponentAnalysis(BasePreprocessor):
    """Principal Component Analysis (PCA)

    A statistical procedure that uses an orthogonal transformation to convert
    a set of observations of possibly correlated variables (entities each of
    which takes on various numerical values) into a set of values of linearly
    uncorrelated variables called principal components.
    
    Parameters:
    -----------
    n_components : integer
    
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

        self.components = None
        self.explained_variance = []
        self.explained_variance_ratio = []

    def fit(self, X):
        X = super().fit(X)

        # Computing the Covariance Matrix
        S = get_covariance_matrix(X)

        # Compute the eigenvalues and eigenvectors
        eigvals, eigvecs = linalg.eig(S)
        eigvals_sum = np.sum(eigvals)

        # Selecting principal components for the new feature subspace
        eig_pairs = [(eigvals[i], eigvecs[:, i]) for i in range(S.shape[0])]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        self.components = np.column_stack((
            eig_pairs[i][1] for i in range(self.n_components)
        ))
        self.explained_variance = np.array([
            eig_pairs[i][0] for i in range(self.n_components)
        ])
        self.explained_variance_ratio = np.array([
            v / eigvals_sum for v in self.explained_variance
        ])

        return self

    def transform(self, X):
        X = super().transform(X)

        # Transforming the samples onto the new subspace
        return X @ self.components
