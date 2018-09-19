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
    n_features = X.shape[1]
    n_classes = np.unique(y).size
    Sw = np.zeros((n_features, n_features))

    for cl, m in zip(range(n_classes), mean_vectors):
        Si = np.zeros((n_features, n_features))
        m = m.reshape(4, 1)

        for x in X[y == cl, :]:
            v = (x.reshape(4, 1) - m)
            Si +=  v @ v.T
        Sw += Si

    return Sw

def get_between_class_scatter_matrix(mean_vectors, X, y):
    n_features = X.shape[1]
    n_classes = np.unique(y).size

    Sb = np.zeros((n_features, n_features))
    m = np.mean(X, axis=0).reshape(4, 1)

    for cl, m_i in zip(range(n_classes), mean_vectors):
        n = X[y == cl, :].shape[0]
        v = (m_i.reshape(4, 1) - m)
        Sb += n * v @ v.T

    return Sb
