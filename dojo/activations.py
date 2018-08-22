import numpy as np

__all__ = [
    "sigmoid",
]

def sigmoid(x):
    z = lambda k: 1.0/(1 + np.exp(-k))

    if type(x) in [list, np.ndarray]:
        for i in range(len(x)):
            x[i] = z(x[i])
        return x

    else:
        return z(x)
