import os

from . import (
    anomaly,
    base,
    bayes,
    cluster,
    dimred,
    ensemble,
    linear,
    metrics,
    preprocessing,
    recommender,
    split,
    svm,
    tree,
    activations,
    exceptions,
)

__all__ = [
    "anomaly",
    "base",
    "bayes",
    "cluster",
    "dimred",
    "ensemble",
    "linear",
    "metrics",
    "preprocessing",
    "recommender",
    "split",
    "svm",
    "tree",
    "activations",
    "exceptions",
]

DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(DIR, "../VERSION"), 'r') as f:
    __version__ = f.read()
