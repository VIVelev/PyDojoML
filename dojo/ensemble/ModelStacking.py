import numpy as np
from ..base import BaseModel

__all__ = [
    "ModelStacking",
]


class ModelStacking(BaseModel):
    # TODO: add __doc__

    def __init__(self, level_one_models, level_two_models):
        self.level_one_models = level_one_models
        self.level_two_models = level_two_models


    def fit(self, X, y):
        new_columns = []
        for i in range(len(self.level_one_models)):
            self.level_one_models[i].fit(X, y)
            new_columns.append(self.level_one_models[i].predict(X))

        X_final = np.hstack((
            X, np.transpose(new_columns)
        ))

        for i in range(len(self.level_two_models)):
            self.level_two_models.fit(X_final, y)

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        pass
