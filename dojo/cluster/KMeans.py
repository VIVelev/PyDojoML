from .utils import np, linalg

__all__ = [
    "KMeans",
]

class KMeans:
    def __init__(self, n_clusters=2, n_init=10):
        self.n_clusters = n_clusters
        self.n_init = n_init

        self.centroids = []
        self.distortion = 0
        self.clusters = []
        self._X = None

    def _calc_distortion(self):
        m = self._X.shape[0]
        return 1/m * sum(
            linalg.norm(self._X[i, :] - self.centroids[self.clusters[i]])**2 for i in range(m)
        )

    def _init_random_centroids(self):
        self.centroids = self._X[np.random.choice(list(range(self._X.shape[0])), size=self.n_clusters), :]

    def _move_centroids(self):
        self.centroids = np.array(
            [np.mean(self._X[self.clusters == k, :], axis=0) for k in range(self.n_clusters)]
        )

    def _closest_centroid(self, x):
        closest_centroid = 0
        distance = 10^9

        for i in range(self.n_clusters):
            current_distance = linalg.norm(x - self.centroids[i])
            if current_distance < distance:
                closest_centroid = i
                distance = current_distance

        return closest_centroid

    def _assign_clusters(self):
        self.clusters = np.array([self._closest_centroid(x) for x in self._X])

    def fit(self, X):
        self._X = X
        candidates = []

        for _ in range(self.n_init):
            self._init_random_centroids()
            while True:
                prev_clusters = self.clusters
                self._assign_clusters()
                self._move_centroids()

                if np.all(prev_clusters == self.clusters):
                    break

            self.distortion = self._calc_distortion()
            candidates.append((self.centroids, self.distortion, self.clusters))
        
        candidates.sort(key=lambda x: x[1])
        self.centroids = candidates[0][0]
        self.distortion = candidates[0][1]
        self.clusters = candidates[0][2]

        return self
