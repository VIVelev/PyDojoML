import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "plot_decision_boundary",
]


def plot_decision_boundary(model, X, y, step=0.1, figsize=(10, 8), alpha=0.4, size=20):
    """Plots the classification decision boundary of `model` on `X` with labels `y`.
    Using numpy and matplotlib.
    """

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    f, ax = plt.subplots(figsize=figsize)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=alpha)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=size, edgecolor='k')
    plt.show()
