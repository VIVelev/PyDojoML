from .utils import (
    np, Preprocessor,

    kl_divergence,
)

__all__ = [
    "TSNE",
]


class TSNE(Preprocessor):
    """t-distributed Stochastic Neighbor Embedding (t-SNE)
    
    t-distributed Stochastic Neighbor Embedding (t-SNE) is a machine learning algorithm for dimensionality reduction
    developed by Geoffrey Hinton and Laurens van der Maaten. It is a nonlinear dimensionality reduction technique
    that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions,
    which can then be visualized in a scatter plot. Specifically, it models each high-dimensional object by a
    two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar
    objects are modeled by distant points.

    For a good user guide on `How to Use t-SNE Effectively`: https://distill.pub/2016/misread-tsne/
    Here is the original paper: http://www.cs.toronto.edu/~hinton/absps/tsne.pdf

    Parameters:
    -----------
    n_components : integer
    perplexity : integer
    learning_rate : float
    momentum : float, (0, 1]
    sigma : float
    n_iter : integer
    verbose : boolean

    """

    def __init__(self, n_components=2, perplexity=30, learning_rate=200.0, momentum=0.99, sigma=1.0, n_iter=1000, verbose=False):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sigma = sigma
        self.n_iter = n_iter
        self.verbose = verbose

    def _high_dim_sim(self, v, w, normalize=False, X=None, idx=None):
        """Similarity measurement based on Gaussian Distribution"""

        sim = np.exp((-np.linalg.norm(v - w) ** 2) / (2*self.sigma ** 2))

        if normalize:
            sum_norm = sum(map(lambda x: x[1], self._knn(idx, X, high_dim=True)))
            return sim / sum_norm
        else:
            return sim

    def _low_dim_sim(self, v, w, normalize=False, Y=None, idx=None):
        """Similarity measurement based on (Student) t-Distribution"""

        sim = (1 + np.linalg.norm(v - w) ** 2) ** -1

        if normalize:
            sum_norm = sum(map(lambda x: x[1], self._knn(idx, Y, high_dim=False)))
            return sim / sum_norm
        else:
            return sim

    def _knn(self, i, X, high_dim=True):
        knns = []
        for j in range(X.shape[0]):
            if j != i:
                if high_dim:
                    distance = self._high_dim_sim(X[i], X[j])
                else:
                    distance = self._low_dim_sim(X[i], X[j])
                knns.append([j, distance])

        return sorted(knns, key=lambda x: x[1])[:self.perplexity]

    def _get_high_dim_dist(self, X):
        table = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i != j:
                    Pij = self._high_dim_sim(X[i], X[j], normalize=True, X=X, idx=i)
                    Pji = self._high_dim_sim(X[i], X[j], normalize=True, X=X, idx=j)
                    table[i, j] = (Pij + Pji) / (2*X.shape[0])

        return table

    def _get_low_dim_dist(self, Y):
        table = np.zeros((Y.shape[0], Y.shape[0]))

        for i in range(Y.shape[0]):
            for j in range(Y.shape[0]):
                if i != j:
                    Pij = self._low_dim_sim(Y[i], Y[j], normalize=True, Y=Y, idx=i)
                    Pji = self._low_dim_sim(Y[i], Y[j], normalize=True, Y=Y, idx=j)
                    table[i, j] = (Pij + Pji) / (2*Y.shape[0])

        return table

    def _gradient(self, high_dim_dist, low_dim_dist, Y, i):
        """Computes the gradient of KL divergence with respect to the i'th example of Y"""

        return 4 * sum([
            (high_dim_dist[i, j] - low_dim_dist[i, j]) * (Y[i] - Y[j]) * self._low_dim_sim(Y[i], Y[j]) \
            for j in range(Y.shape[0])
        ])

    def fit(self, X):
        """Gradient Descent optimization process"""

        # compute high-dimensional affinities (Gaussian Distribution)
        high_dim_dist = self._get_high_dim_dist(X)
        # Sample initial solutions
        Y = np.random.randn(X.shape[0], self.n_components)

        prev_Ys = [Y, Y]

        for iteration in range(1, self.n_iter+1):
            # compute low-dimensional affinities (Student t-Distribution)
            low_dim_dist = self._get_low_dim_dist(Y)
    
            for i in range(Y.shape[0]):
                # compute gradient
                grad = self._gradient(high_dim_dist, low_dim_dist, Y, i)
                # set new Y[i]
                Y[i] = prev_Ys[1][i] + self.learning_rate * grad + self.momentum * (prev_Ys[1][i] - prev_Ys[0][i])

            low_dim_dist = self._get_low_dim_dist(Y)
            prev_Ys = [prev_Ys[1], Y]

            if iteration % 100 == 0 and self.verbose:
                print(f"ITERATION: {iteration}{3*' '}|||{3*' '}KL divergence: {kl_divergence(high_dim_dist, low_dim_dist)}")

        self.embeddings = Y
        return self

    def transform(self, X):
        return self.embeddings
