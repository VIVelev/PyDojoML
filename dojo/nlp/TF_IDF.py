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
        - raw count; ("raw")
        - frequency - raw count / total number of terms; ("freq")
        - log normalization - log(1 + raw count); ("log")
        - k normalization - k + (1-k)*(raw count / max term count),
        k = 0.5 by default; ("k-norm")

    idf_weighting_scheme : string, optional
    Supported weighting schemes:
        - frequency - log(N / (nt+1e-18)); ("freq")
        - smooth frequency - log(1 + N / (nt+1e-18)); ("smooth")
    
    """

    def __init__(self, tf_weighting_scheme="freq", idf_weighting_scheme="freq", k=0.5):
        self.tf_weighting_scheme = tf_weighting_scheme.lower()
        self.idf_weighting_scheme = idf_weighting_scheme.lower()
        self.k = k

        self.unique_terms = []
        self.idfs = {}

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
            return np.log(N / (nt+1e-18))
        else: # smooth
            return np.log(1 + N / (nt+1e-18))

    def fit(self, X):
        self.unique_terms = np.unique([term for doc in X for term in doc.split(' ')])
        self.idfs = {term: self._inverse_doc_frequency(term, X) for term in self.unique_terms}

        return self

    def transform(self, X):
        return np.array([
            [self._term_frequency(term, doc) * self.idfs[term] for term in self.unique_terms] for doc in X
        ])
