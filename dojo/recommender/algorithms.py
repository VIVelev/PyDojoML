import numpy as np
from .users import CBUser

__all__ = [
    "ContentBased",
    "CollaborativeFiltering",
]


class ContentBased:
    # TODO: Add __doc__

    def __init__(self):
        self.users = []

    def fit(self, ratings):
        ratings = np.array(ratings)
        self.users = [
            CBUser(ratings[:, i]) for i in range(ratings.shape[1]) 
        ]
        
        return self

    def recommend(self, user_id=None):
        if user_id is None:
            return [
                user.recommend() for user in self.users
            ]
        else:
            return self.users[user_id].recommend()

class CollaborativeFiltering:
    # TODO: Add __doc__

    def __init__(self):
        pass

    def fit(self):
        pass

    def recommend(self):
        pass
