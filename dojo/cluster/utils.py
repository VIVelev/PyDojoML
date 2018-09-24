import numpy as np
from scipy import linalg

from scipy.cluster.hierarchy import (
    linkage,
    fcluster,
    dendrogram,
)

from ..base import BaseClustering
from ..exceptions import ParameterError
