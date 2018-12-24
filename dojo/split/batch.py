import numpy as np

__all__ = [
    "batch_iterator",
]


class batch_iterator:

    def __init__(self, X, y, batch_size=32):
        rnd_idxs = np.random.permutation(list(range(X.shape[0])))
        self.X = X[rnd_idxs]
        self.y = y[rnd_idxs]
        self.batch_size = batch_size
        self.i = -self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        self.i += self.batch_size
        if self.i >= self.X.shape[0]:
            raise StopIteration

        return (self.X[self.i:self.i+self.batch_size],
            self.y[self.i:self.i+self.batch_size])
