import numpy as np
from scipy import linalg

from ..base import BaseModel
from ..activations import sigmoid

from ..exceptions import MethodNotSupportedError

from ..metrics.regression import mean_squared_error
from ..metrics.classification import accuracy_score
