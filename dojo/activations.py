import numpy as np

__all__ = [
    "sigmoid",
]

def sigmoid(x):
    """The Sigmoid (Logistic) function.

    s(k) = 1.0 / (1.0 + e^-k)
    
    Parameters:
    -----------
    x : any real number or a vector of real numbers
    
    Returns:
    --------
    res : float number or a vector of float numbers
    The output from the sigmoid function.
    
    """

    z = lambda k: 1.0/(1 + np.exp(-k))

    if type(x) in [list, np.ndarray]:
        for i in range(len(x)):
            x[i] = z(x[i])
        return x

    else:
        return z(x)
