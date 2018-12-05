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
            data = np.column_stack((
                self.X, self.y
            ))
            np.random.shuffle(data)

        self.X, self.y = data[:, :-1], data[:, -1]
        
        self._i = 0
        self._X_folds = np.split(self.X, k, axis=0)
        self._y_folds = np.split(self.y, k)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self.k:
            raise StopIteration
        
        X_train = np.zeros((1, self.X.shape[1]))
        X_test = y_train = y_test = []

        for j in range(self.k):
            if j == self._i:
                X_test = self._X_folds[self._i]
                y_test = self._y_folds[self._i]
            else:
                X_train = np.vstack((
                    X_train, self._X_folds[j]
                ))
                y_train = np.append(y_train, self._y_folds[j])
        
        self._i+=1
        return X_train[1:, :], X_test, y_train, y_test
