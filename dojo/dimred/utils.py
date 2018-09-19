import numpy as np
from scipy import linalg

__all__ = [
    "get_mean_vectors",
    "get_within_class_scatter_matrix",
    "get_between_class_scatter_matrix",
]

def get_mean_vectors(X, y):
    return [np.mean(X[y == cl, :], axis=0) for cl in y]

def get_within_class_scatter_matrix(mean_vectors, X, y):
    d = X.shape[1]
    c = np.unique(y).size
    Sw = np.zeros(d, d)

    for cl, m in zip(range(c), mean_vectors):
        Si = np.zeros(d, d)
        for x in X[y == cl, :]:
            Si += (x - m) @ (x - m).T
        Sw += Si
    
    return Sw

def get_between_class_scatter_matrix(mean_vectors, X, y):
    d = X.shape[1]
    c = np.unique(y).size

    m = np.mean(X, axis=0)
    Sb = np.zeros(d, d)

    for cl, m_i in zip(range(c), mean_vectors):
        n = X[y == cl, :].shape[0]
        Sb += n * (m_i - m) @ (m_i - m).T

    return Sb
