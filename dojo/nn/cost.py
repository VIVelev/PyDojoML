import numpy as np

__all__ = [
    "cross_entropy",
]


def cross_entropy(Y, AL):
    # Avoid division by zero
    AL = np.clip(AL, 1e-18, 1-1e-18)

    # Cross-Entropy Cost function
    return np.squeeze(
        - np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    )
