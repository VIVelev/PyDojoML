import numpy as np

__all__ = [
    "linear_forward",
    "linear_activation_forward",
    "forward_prop",
]


def linear_forward(A, W, b):
    return W @ A + b

def linear_activation_forward(A, W, b, activation):
    pass

def forward_prop(A0, L, parameters):
    pass
