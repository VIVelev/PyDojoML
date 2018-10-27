import numpy as np
from ..linear import LinearRegression

class User:
    def __init__(self, id, data):
        self.id = id
        self.data = np.array(data)

        self.model = None

    def fit(self):
        self.model = LinearRegression().fit(self.data[:, :-1], self.data[:, -1])

    def predict(self, x):
        return self.model.predict([x])[0]

class ContentBased:
    def __init__(self):
        self.users = []

    def fit(self, X):
        pass

    def recommend(self, user_id=None):
        pass
