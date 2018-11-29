import numpy as np

from ..base import BaseModel
from ..metrics import accuracy_score

__all__ = [
    "ModelStacking",
]


class ModelStacking(BaseModel):
    # TODO: add __doc__

    def __init__(self, first_level_models, second_level_model):
        self.first_level_models = first_level_models
        self.second_level_model = second_level_model

    def fit(self, X, y):
        X, y = super().fit(X, y)

        new_columns = []
        for i in range(len(self.first_level_models)):
            self.first_level_models[i].fit(X, y)
            new_columns.append(self.first_level_models[i].predict(X))

        X_final = np.hstack((
            X, np.transpose(new_columns)
        ))
        self.second_level_model.fit(X_final, y)
        return self

    def _prepare_data(self, X):
        new_columns = []
        for model in self.first_level_models:
            new_columns.append(model.predict(X))

        return np.hstack((
            X, np.transpose(new_columns)
        ))

    def predict(self, X):
        return self.second_level_model.predict(self._prepare_data(X))

    def predict_proba(self, X):
        return self.second_level_model.predict_proba(self._prepare_data(X))

    def decision_function(self, X):
        return self.second_level_model.decision_function(self._prepare_data(X))

    def evaluate(self, X, y):
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )
