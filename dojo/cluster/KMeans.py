from .utils import (
    np,
    linalg,

    BaseClustering,
)

__all__ = [
    "KMeans",
]

class KMeans(BaseClustering):
    """K-Means Clustering algorithm
    
    Parameters:
    -----------
    n_clusters : integer, optional
    n_runs : integer, how many times to run the algorithm, optional
    
    """

    def __init__(self, n_clusters=2, n_runs=10):
        self.n_clusters = n_clusters
        self.n_runs = n_runs

        self.distortion = 0
        self.centroids = []
        self.clusters = []
        self._X = None

    def _calc_distortion(self):
        """Calculates the distortion value of the current clusters
        """
        m = self._X.shape[0]
        self.distortion = 1/m * sum(
            linalg.norm(self._X[i, :] - self.centroids[self.clusters[i]])**2 for i in range(m)
        )
        return self.distortion

    def _init_random_centroids(self):
        """Initialize the centroids as k random samples of X (k = n_clusters)
        """
        self.centroids = self._X[np.random.choice(list(range(self._X.shape[0])), size=self.n_clusters), :]

    def _move_centroids(self):
        """Calculate new centroids as the means of the samples in each cluster
        """
        self.centroids = np.array([
            np.mean(self._X[self.clusters == k, :], axis=0) for k in range(self.n_clusters)
        ])

    def _closest_centroid(self, x):
        """Returns the index of the closest centroid to the sample
        """
        closest_centroid = 0
        distance = 10^9

        for i in range(self.n_clusters):
            current_distance = linalg.norm(x - self.centroids[i])
            if current_distance < distance:
                closest_centroid = i
                distance = current_distance

        return closest_centroid

    def _assign_clusters(self):
        """Assign the samples to the closest centroids to create clusters
        """
        self.clusters = np.array([self._closest_centroid(x) for x in self._X])

    def fit(self, X):
        """The K-Means itself
        """

        self._X = super().cluster(X)
        candidates = []

        for _ in range(self.n_runs):
            self._init_random_centroids()
            while True:
                prev_clusters = self.clusters
                self._assign_clusters()
                self._move_centroids()

                if np.all(prev_clusters == self.clusters):
                    break

            self._calc_distortion()
            candidates.append((self.distortion, self.centroids, self.clusters))
        
        candidates.sort(key=lambda x: x[0])
        self.distortion = candidates[0][0]
        self.centroids = candidates[0][1]
        self.clusters = candidates[0][2]

        return self

    def cluster(self, X):
        X = super().cluster(X)
        return np.array([self._closest_centroid(x) for x in X])
