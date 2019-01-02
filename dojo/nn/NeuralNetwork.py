import numpy as np
import progressbar
from terminaltables import AsciiTable

from ..base import BaseModel
from ..losses import CrossEntropy
from ..metrics.classification import accuracy_score
from ..misc import bar_widgets
from ..optimizers import Adam
from ..preprocessing import OneHotEncoder
from ..split import batch_iterator
from .layers import ActivationLayer, Dense

__all__ = [
    "NeuralNetwork",
]


class NeuralNetwork(BaseModel):
    # TODO: add __doc__

    def __init__(self, optimizer=Adam(0.01), n_epochs=5_000, batch_size=32, loss=CrossEntropy(), verbose=False):
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose

        self._progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        self._loss_values = []
        self._layers = []

    def add(self, layer):
        if len(self._layers) > 0:
            layer.n_inputs = self._layers[-1].n_neurons

            # For activation layers n_neurons equals the n_inputs
            if isinstance(layer, ActivationLayer):
                layer.n_neurons = layer.n_inputs

        layer.init_weights()
        self._layers.append(layer)

    def forwardprop(self, X):
        AL = X
        for layer in self._layers:
            AL = layer.forward(AL)

        return AL

    def backprop(self, Y, AL):
        accum_grad = self.loss.gradient(Y, AL)
        for layer in reversed(self._layers):
            accum_grad = layer.backward(accum_grad)

    def train_on_batch(self, X, y):
        X, y = X.T, y.T
        assert X.shape[1] == y.shape[1]

        # Forward-propagation
        AL = self.forwardprop(X)

        # Computing the cost
        self._loss_values.append(np.mean(self.loss(y, AL)))
        penalty = 0
        for layer in self._layers:
            try:
                penalty += layer.regularizer(layer.W)
            except AttributeError:
                pass
        self._loss_values[-1] += 1/X.shape[1] * penalty

        # Back-propagation
        self.backprop(y, AL)

        # Updating the weights
        for layer in self._layers:
            layer.update(self.optimizer)

        return self._loss_values[-1]

    def fit(self, X, y):
        X, y = super().fit(X, y)

        for n_epoch in self._progressbar(range(1, self.n_epochs+1)):
            for X_batch, y_batch in batch_iterator(X, OneHotEncoder().fit_transform(y), batch_size=self.batch_size):
                self.train_on_batch(X_batch, y_batch)

            # Printing
            if n_epoch % 100 == 0 and self.verbose:
                print(f"Epoch {n_epoch}, Cost: {self._loss_values[-1]}")

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)

    def predict_proba(self, X):
        X = super().predict_proba(X).T
        return self.forwardprop(X)

    def decision_function(self, X):
        pass

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))

    def summary(self, name="Model Summary"):
        # Print model name
        print(AsciiTable([[name]]).table)
        # Network input shape (first layer's input shape)
        print("Input Shape: %s" % str((self._layers[0].n_inputs, 1)))
        # Iterate through network and get each layer's configuration
        table_data = [["Layer Type", "Number of Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self._layers:
            layer_type = layer.get_name()
            n_params = layer.get_n_params()
            output_shape = (layer.n_neurons, 1)
            table_data.append([layer_type, str(n_params), str(output_shape)])
            tot_params += n_params
        # Print network configuration table
        print(AsciiTable(table_data).table)
        print("Total Number of Parameters: %d\n" % tot_params)
