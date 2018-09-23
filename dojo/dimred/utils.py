import numpy as np
from scipy import linalg

from ..base import BasePreprocessor

__all__ = [
    "get_mean_vectors",
    "get_within_class_scatter_matrix",
    "get_between_class_scatter_matrix",
    "get_covariance_matrix",
]

def get_mean_vectors(X, y):
    """Calculates the mean samples per class
    
    Parameters:
    -----------
    X : array-like, shape (m, n) - the samples
    y : array-like, shape (m, ) - the class labels
    
    Returns:
    --------
    mean_vectors : array-like, shape (k, )
    Those are the mean samples from each k classes.
    
    """

    return [np.mean(X[y == cl, :], axis=0) for cl in np.unique(y)]

def get_within_class_scatter_matrix(X, y):
    """Calculates the Within-Class Scatter matrix
    
    Parameters:
    -----------
    X : array-like, shape (m, n) - the samples
    y : array-like, shape (m, ) - the class labels
    
    Returns:
    --------
    within_class_scatter_matrix : array-like, shape (n, n)
    
    """

    mean_vectors = get_mean_vectors(X, y)
    n_features = X.shape[1]
    Sw = np.zeros((n_features, n_features))

    for cl, m in zip(np.unique(y), mean_vectors):
        Si = np.zeros((n_features, n_features))
        m = m.reshape(n_features, 1)

        for x in X[y == cl, :]:
            v = x.reshape(n_features, 1) - m
            Si += v @ v.T
        Sw += Si

    return Sw

def get_between_class_scatter_matrix(X, y):
    """Calculates the Between-Class Scatter matrix
    
    Parameters:
    -----------
    X : array-like, shape (m, n) - the samples
    y : array-like, shape (m, ) - the class labels

    Returns:
    --------
    between_class_scatter_matrix : array-like, shape (n, n)
    
    """

    mean_vectors = get_mean_vectors(X, y)
    n_features = X.shape[1]
    Sb = np.zeros((n_features, n_features))
    m = np.mean(X, axis=0).reshape(n_features, 1)

    for cl, m_i in zip(np.unique(y), mean_vectors):
        v = m_i.reshape(n_features, 1) - m
        Sb += X[y == cl, :].shape[0] * v @ v.T

    return Sb

def get_covariance_matrix(X):
    """Calculates the Variance-Covariance matrix
    
    Parameters:
    -----------
    X : array-like, shape (m, n) - the data
    
    Returns:
    --------
    variance_covariance_matrix : array-like, shape(n, n)
    
    """

    n_features = X.shape[1]    
    S = np.zeros((n_features, n_features))
    m = np.mean(X, axis=0).reshape(n_features, 1)

    for x in X:
        v = x.reshape(n_features, 1) - m
        S += v @ v.T

    return 1/(X.shape[0]-1) * S
