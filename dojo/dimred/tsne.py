from .utils import np, Preprocessor

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

    def _higher_dim_sim(self, v, w, normalize=False, X=None, idx=None):
        """Computes a Gaussian Distribution"""

        sim = np.exp((-np.linalg.norm(v - w) ** 2) / (2*self.sigma ** 2))

        if normalize:
            sum_norm = sum(map(lambda x: x[1], self._knn(idx, X, higher_dim=True)))
            return sim / sum_norm
        else:
            return sim

    def _lower_dim_sim(self, v, w, normalize=False, Y=None, idx=None):
        """Computes a (Student's) t-Distribution"""

        sim = (1 + np.linalg.norm(v - w) ** 2) ** -1

        if normalize:
            sum_norm = sum(map(lambda x: x[1], self._knn(idx, Y, higher_dim=False)))
            return sim / sum_norm
        else:
            return sim

    def _knn(self, i, X, higher_dim=True):
        knns = []
        for j in range(X.shape[0]):
            if j != i:
                if higher_dim:
                    distance = self._higher_dim_sim(X[i], X[j], normalize=False)
                else:
                    distance = self._lower_dim_sim(X[i], X[j], normalize=False)
                knns.append([j, distance])

        return sorted(knns, key=lambda x: x[1])[:self.perplexity]

    def _get_higher_dim_dist(self, X):
        table = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i != j:
                    Pij = self._higher_dim_sim(X[i], X[j], normalize=True, X=X, idx=i)
                    Pji = self._higher_dim_sim(X[i], X[j], normalize=True, X=X, idx=j)
                    table[i, j] = (Pij + Pji) / (2*X.shape[0])

        return table

    def _get_lower_dim_dist(self, Y):
        table = np.zeros((Y.shape[0], Y.shape[0]))

        for i in range(Y.shape[0]):
            for j in range(Y.shape[0]):
                if i != j:
                    Pij = self._lower_dim_sim(Y[i], Y[j], normalize=True, Y=Y, idx=i)
                    Pji = self._lower_dim_sim(Y[i], Y[j], normalize=True, Y=Y, idx=j)
                    table[i, j] = (Pij + Pji) / (2*Y.shape[0])

        return table

    def _kl_divergence(self, higher_dim_dist, lower_dim_dist):
        """Symmentric Kullback–Leibler divergence (KL divergence)
        
        A measure of how one probability distribution is different from a second, reference probability distribution.
        Wikipedia page: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        
        """

        cost = 0
        for i in range(higher_dim_dist.shape[0]):
            for j in range(lower_dim_dist.shape[0]):
                if higher_dim_dist[i, j] != 0 and lower_dim_dist[i, j] != 0:
                    cost += higher_dim_dist[i, j] * np.log( higher_dim_dist[i, j] / lower_dim_dist[i, j] )
        
        return cost

    def _gradient(self, higher_dim_dist, lower_dim_dist, Y, i):
        return 4 * sum([
            (higher_dim_dist[i, j] - lower_dim_dist[i, j]) * (Y[i] - Y[j]) * self._lower_dim_sim(Y[i], Y[j]) \
            for j in range(Y.shape[0])
        ])

    def _optimize(self, higher_dim_dist, lower_dim_dist, Y):
        prev_Ys = [Y]*2

        for iteration in range(1, self.n_iter+1):
            for i in range(Y.shape[0]):
                grad = self._gradient(higher_dim_dist, lower_dim_dist, Y, i) # Gradient
                Y[i] -= self.learning_rate * grad + self.momentum * (prev_Ys[1][i] - prev_Ys[0][i])

            prev_Ys = [prev_Ys[1], Y]

            if iteration % 100 == 0 and self.verbose:
                print(f"ITERATION: {iteration}{3*' '}|||{3*' '}KL divergence: {self._kl_divergence(higher_dim_dist, self._get_lower_dim_dist(Y))}")

        return Y

    def fit(self, X):
        pass

    def transform(self, X):
        Y = np.random.randn(X.shape[0], self.n_components)

        return self._optimize(
            self._get_higher_dim_dist(X),
            self._get_lower_dim_dist(Y),
            Y
        )
