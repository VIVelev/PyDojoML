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
        
        self.none_idxs = []
        self.non_rated = []
        self.model = None

        self.fit(ratings)

    def fit(self, ratings):
        ratings = np.array(ratings)

        self.none_idxs = np.where(ratings == None)
        self.none_idxs = [self.none_idxs[i][0] for i in range(len(self.none_idxs))]
        self.non_rated = [ratings[i, :] for i in self.none_idxs]

        ratings = [ratings[i, :] for i in range(ratings.shape[0]) if not i in self.none_idxs]
        X, y = ratings[:, :-1], ratings[:, -1]
        self.model = LinearRegression().fit(X, y)

        return self

    def recommend(self):
        pred = self.model.predict(self.non_rated)
        return {
            self.none_idxs[i]: pred[i] for i in range(len(self.none_idxs))
        }
