import numpy as np

from ..base import BaseModel
from ..metrics import accuracy_score

__all__ = [
    "ModelStacking",
]


class ModelStacking(BaseModel):
    # TODO: add __doc__

    def __init__(self, level_one_models, level_two_model):
        self.level_one_models = level_one_models
        self.level_two_model = level_two_model

    def fit(self, X, y):
        X, y = super().fit(X, y)

        new_columns = []
        for i in range(len(self.level_one_models)):
            self.level_one_models[i].fit(X, y)
            new_columns.append(self.level_one_models[i].predict(X))

        X_final = np.hstack((
            X, np.transpose(new_columns)
        ))
        self.level_two_model.fit(X_final, y)
        return self

    def _prepare_data(self, X):
        new_columns = []
        for model in self.level_one_models:
            new_columns.append(model.predict(X))

        return np.hstack((
            X, np.transpose(new_columns)
        ))

    def predict(self, X):
        return self.level_two_model.predict(self._prepare_data(X))

    def predict_proba(self, X):
        return self.level_two_model.predict_proba(self._prepare_data(X))

    def decision_function(self, X):
        return self.level_two_model.decision_function(self._prepare_data(X))

    def evaluate(self, X, y):
        print(
            f"Accuracy score: {accuracy_score(y, self.predict(X))}"
        )
