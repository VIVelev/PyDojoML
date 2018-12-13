import numpy as np

__all__ = [
    "cross_entropy",
]


def cross_entropy(Y, AL):
    return np.squeeze(
        - np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    )
