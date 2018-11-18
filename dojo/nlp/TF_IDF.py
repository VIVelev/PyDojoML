import numpy as np
from ..base import BasePreprocessor

__all__ = [
    "TF_IDF",
]


class TF_IDF(BasePreprocessor):
    """Term Frequency - Inverse Document Frequency
    Text Vectorization.
    
    Parameters:
    -----------
    tf_weighting_scheme : string, optional
    Supported weighting schemes:
        - raw count ("raw")
        - frequency ("freq")
        - log normalization ("log")
        - k normalization, k = 0.5 by default ("k-norm")

    idf_weighting_scheme : string, optional
    Supported weighting schemes:
        - frequency ("freq")
        - smooth frequency ("smooth")
    
    """

    def __init__(self, tf_weighting_scheme="freq", idf_weighting_scheme="freq", k=0.5):
        self.tf_weighting_scheme = tf_weighting_scheme.lower()
        self.idf_weighting_scheme = idf_weighting_scheme.lower()
        self.k = k

    def _term_frequency(self, t, d):
        arr = np.array(d.split(' '))
        raw_count = np.count_nonzero(
            arr == t
        )

        if self.tf_weighting_scheme == "raw":
            return raw_count
        elif self.tf_weighting_scheme == "freq":
            return raw_count / arr.size
        elif self.tf_weighting_scheme == "log":
            return np.log(1+raw_count)
        else: # k-norm
            return self.k + (1-self.k) * raw_count / max(list(arr).count(x) for x in np.unique(arr))

    def _inverse_doc_frequency(self, t, D):
        N = len(D)
        nt = len([1 for doc in D if t in doc])

        if self.idf_weighting_scheme == "freq":
            return np.log(N / (nt+1))
        else: # smooth
            return np.log(1 + N / (nt+1))

    def fit(self, X):
        pass

    def transform(self, X):
        pass
