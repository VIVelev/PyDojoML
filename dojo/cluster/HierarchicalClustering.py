from .utils import (
    np,
    linkage,
    fcluster,
    dendrogram,

    BaseClustering,
    ParameterError,
)

__all__ = [
    "HierarchicalClustering",
]

class HierarchicalClustering(BaseClustering):
    """Agglomerative Hierarchical Clustering
    
    Parameters:
    -----------
    mode : string, optional
        - "n_clusters"
        - "max_distance"
    n_clusters : integer, optional
    max_distance : float, optional
    linkage : string, optional
        - "single"
        - "complete"
        - "average"
        - "centroid"
        - "ward"
    
    """

    def __init__(self, mode="n_clusters", n_clusters=2, max_distance=None, linkage="ward"):
        self.mode = mode
        self.n_clusters = n_clusters
        self.max_distance = max_distance
        self.linkage = linkage

    def cluster(self, X):
        X = super().cluster(X)
        self._distances = linkage(X, method=self.linkage)

        if self.mode == "n_clusters":
            return fcluster(
                self._distances,
                self.n_clusters,
                criterion="maxclust"
            )

        elif self.mode == "max_distance":
            return fcluster(
                self._distances,
                self.max_distance,
                criterion="distance"
            )

        else:
            raise ParameterError(f"Unknown / unsupported clustering mode: \"{self.mode}\"")

    def plot_dendrogram(self):
        dendrogram(self._distances)
