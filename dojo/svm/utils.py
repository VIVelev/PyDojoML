from ..base import BaseModel
from ..exceptions import MethodNotSupportedError

from .libsvm.svmutil import (
    svm_problem,
    svm_parameter,
    svm_train,
    svm_predict,
)

from ..metrics.classification import accuracy_score
from ..metrics.regression import mean_squared_error

def set_kernel(kernel_string):
    if kernel_string.upper() == "LINEAR":
        return 0
    elif kernel_string.upper() == "POLY":
        return 1
    elif kernel_string.upper() == "SIGMOID":
        return 3
    else:
        return 2 # RBF kernel
