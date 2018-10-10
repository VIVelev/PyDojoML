import numpy as np

__all__ = [
    "KFolds",
]

class KFolds:
    def __init__(self, X, y, k=5):
        self.X = X
        self.y = y

        while len(X) % k != 0:
            k-=1
        self.k = k
        
        self.i = 0
        self._X_folds = np.split(X, k, axis=0)
        self._y_folds = np.split(y, k)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.k:
            raise StopIteration

        X_train = X_test = y_train = y_test = np.array([])

        for j in range(self.k):
            if j == self.i:
                X_test = self._X_folds[self.i]
                y_test = self._y_folds[self.i]
            else:
                X_train = np.vstack((
                    X_train, self._X_folds[j]
                ))
                y_train = np.append(y_train, self._y_folds[j])

        return X_train, X_test, y_train, y_test
