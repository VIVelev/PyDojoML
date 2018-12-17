import numpy as np

__all__ = [
    "L2",
]


class L2:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, W):
        return (self.lambd / 2) * np.sum(np.square(W))

    def gradient(self, W):
        return self.lambd * W
