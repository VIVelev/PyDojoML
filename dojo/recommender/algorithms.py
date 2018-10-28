import numpy as np
from .users import User

__all__ = [
    "ContentBased",
]

class ContentBased:
    def __init__(self):
        self.users = []

    def fit(self, ratings):
        ratings = np.array(ratings)

        self.users = [
            User(ratings[:, i]) for i in range(ratings.shape[1]) 
        ]

    def recommend(self, user_id=None):
        if user_id is None:
            return [
                user.recommend() for user in self.users
            ]
        else:
            return self.users[user_id].recommend()
