import numpy as np

__all__ = [
    "initialize_parameters",
]


def initialize_parameters(layers_dims):
    parameters = {}

    for l in range(1, len(layers_dims)):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters
