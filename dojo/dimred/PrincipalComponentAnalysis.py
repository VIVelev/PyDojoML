from .utils import (
    np, linalg,
    BasePreprocessor,
)

__all__ = [
    "PrincipalComponentAnalysis",
]

class PrincipalComponentAnalysis(BasePreprocessor):

    def __init__(self, n_components=None):
        self.n_components = n_components
        self._W = None

    def fit(self, X):
        # Step 1: Computing the d-dimensional mean vector
        m = np.mean(X, axis=0)

        # Step 2: Computing the Covariance Matrix
        S = np.cov(X)

        # Step 3: Compute the eigenvalues and eigenvectors
        eigvals, eigvecs = linalg.eig(S)

        # Step 4: Selecting principal components for the new feature subspace
        eig_pairs = [(eigvals[i], eigvecs[:, i]) for i in range(S.shape[0])]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        self._W = np.column_stack((
            eig_pairs[i][1] for i in range(self.n_components)
        ))

        return self

    def transform(self, X):
        X = super().transform(X)

        # Step 5: Transforming the samples onto the new subspace
        return X @ self._W
