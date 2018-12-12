import numpy as np

__all__ = [
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",

    "sigmoid_grad",
    "tanh_grad",
    "relu_grad",
    "leaky_relu_grad",
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

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.vectorize(max)(0, x)

def leaky_relu(x, eps=0.01):
    return np.vectorize(max)(0.01*x, x)

# ====================================================================================================
# ====================================================================================================

def sigmoid_grad(x):
    a = sigmoid(x)
    return  a * (1 - a)

def tanh_grad(x):
    return 1 - tanh(x)**2

def relu_grad(x):
    if x >= 0:
        return 1
    else:
        return 0

def leaky_relu_grad(x, eps=0.01):
    if x >= 0:
        return 1
    else:
        return eps
