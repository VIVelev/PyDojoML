from .utils import (
    np, linalg,

    get_mean_vectors,
    get_within_class_scatter_matrix,
    get_between_class_scatter_matrix,
)

__all__ = [
    "LinearDiscriminantAnalysis",
]

class LinearDiscriminantAnalysis:
    """Linear discriminant analysis (LDA)

    A generalization of Fisher's linear discriminant, a method usedin statistics,
    pattern recognition and machine learning to find a linear combination of
    features that characterizes or separates two or more classes of objects or events.
    The resulting combination may be used as a linear classifier, or, more commonly,
    for dimensionality reduction before later classification.
    
    Parameters:
    -----------
    n_components: integer
    
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self._W = None

    def fit(self, X, y):

        # Step 1: Computing the d-dimensional mean vector per class
        mean_vectors = get_mean_vectors(X, y)

        # Step 2: Computing the Scatter Matrices (within-class and between-class)
        Sw = get_within_class_scatter_matrix(mean_vectors, X, y)
        Sb = get_between_class_scatter_matrix(mean_vectors, X, y)
        
        # Step 3: Compute the eigenvalues and eigenvectors of `inv(Sw)@Sb`
        A = linalg.inv(Sw) @ Sb
        eigvals, eigvecs = linalg.eig(A)

        # Step 4: Selecting linear discriminants for the new feature subspace
        eig_pairs = [(eigvals[i], eigvecs[:, i]) for i in range(A.shape[0])]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        self._W = np.column_stack((
            eig_pairs[i][1] for i in range(self.n_components)
        ))

        return self

    def transform(self, X):
        # Step 5: Transforming the samples onto the new subspace
        return X @ self._W
