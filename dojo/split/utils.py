import numpy as np

def convert_assert(X, y):
    X, y = np.array(X), np.array(y)
    assert X.shape[0] == y.shape[0]

    return X, y
