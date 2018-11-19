import numpy as np
from ..linear import LinearRegression

__all__ = [
    "CBUser",
]


class CBUser:
    # TODO: Add __doc__
    
    _id_counter = 0

    def __init__(self, ratings):
        self._id = CBUser._id_counter
        CBUser._id_counter += 1
        
        self._none_idxs = []
        self._non_rated = []
        self._model = None

        self.fit(ratings)

    def fit(self, ratings):
        ratings = np.array(ratings)

        self._none_idxs = np.where(ratings == None)
        self._none_idxs = [self._none_idxs[i][0] for i in range(len(self._none_idxs))]
        self._non_rated = [ratings[i, :] for i in self._none_idxs]

        ratings = [ratings[i, :] for i in range(ratings.shape[0]) if not i in self._none_idxs]
        X, y = ratings[:, :-1], ratings[:, -1]
        self._model = LinearRegression().fit(X, y)

        return self

    def recommend(self):
        pred = self._model.predict(self._non_rated)
        return {
            self._none_idxs[i]: pred[i] for i in range(len(self._none_idxs))
        }
