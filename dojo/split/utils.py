import numpy as np

def convert_assert(X, y):
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    assert X.shape[0] == y.shape[0]

    return X, y
