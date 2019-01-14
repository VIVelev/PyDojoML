from .utils import np, convert_assert

__all__ = [
    "KFolds",
]


class KFolds:
    """K-Folds
    
    Represents the k folds (chunks) used for the
    Cross Validation procedure.
    
    Parameters:
    -----------
    X : matrix, shape (n_samples, n_features)
    y : vector, shape (n_samples, )
    k : integer, optional, the number of folds
    shuffle : boolean, whether to shuffle the data before
    splitting it or not
    
    """

    def __init__(self, X, y, k=5, shuffle=True):
        self.X, self.y = convert_assert(X, y)

        while self.X.shape[0] % k != 0:
            k-=1
        self.k = k

        if shuffle:
            rnd_idxs = np.random.permutation(list(range(X.shape[0])))
            self.X, self.y = self.X[rnd_idxs], self.y[rnd_idxs]
        
        self.test_set_idx = 0
        self.X_folds = np.split(self.X, k, axis=0)
        self.y_folds = np.split(self.y, k)

    def __iter__(self):
        return self

    def __next__(self):
        if self.test_set_idx >= self.k:
            raise StopIteration
        
        X_train = np.zeros((1, self.X.shape[1]))
        X_test = y_train = y_test = []

        for i in range(self.k):
            if i == self.test_set_idx:
                X_test = self.X_folds[self.test_set_idx]
                y_test = self.y_folds[self.test_set_idx]
            else:
                X_train = np.vstack((
                    X_train, self.X_folds[i]
                ))
                y_train = np.append(y_train, self.y_folds[i])
        
        self.test_set_idx += 1
        return X_train[1:, :], X_test, y_train, y_test
