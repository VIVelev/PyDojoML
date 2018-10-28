from .utils import np, convert_assert

__all__ = [
    "train_test_split",
]


def train_test_split(X, y, test_size=0.3, shuffle=True):
    """Train-test split
    
    Splits the data into a training and testing set.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
    The samples.
    y : array-like, shape (n_samples, )
    The labels.
    test_size : float, from 0 to 1
    The proportion of the data that is going to be
    used for testing.
    shuffle : boolean, whether to shuffle the data before
    splitting it or not
    
    Returns:
    --------
    X_train : train-set, samples
    y_train : train-set, labels
    X_test : test-set, samples
    y_test : test-set, labels
    
    """

    X, y = convert_assert(X, y)

    if shuffle:
        data = np.column_stack((
            X, y
        ))

        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1]

    border = int((1 - test_size) * X.shape[0])
    return X[:border, :], y[:border], X[border:, :], y[border:]
