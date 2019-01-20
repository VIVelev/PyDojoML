import numpy as np
from scipy import linalg

from ..base import Classifier, Regressor
from ..activations import Sigmoid
from ..losses import CrossEntropy
from ..regularizers import L2

from ..exceptions import MethodNotSupportedError

from ..metrics.regression import mean_squared_error
from ..metrics.classification import accuracy_score
