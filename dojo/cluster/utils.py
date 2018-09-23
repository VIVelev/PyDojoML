import numpy as np
from scipy.cluster.hierarchy import (
    linkage,
    fcluster,
    dendrogram,
)

from ..exceptions import ParameterError
