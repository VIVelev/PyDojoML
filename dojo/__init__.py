import os

from . import (
    anomaly,
    base,
    cluster,
    dimred,
    ensemble,
    linear,
    metrics,
    preprocessing,
    split,
    svm,
    tree,
    activations,
    exceptions,
)

__all__ = [
    "anomaly",
    "base",
    "cluster",
    "dimred",
    "ensemble",
    "linear",
    "metrics",
    "preprocessing",
    "split",
    "svm",
    "tree",
    "activations",
    "exceptions",
]

DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(DIR, "../VERSION"), 'r') as f:
    __version__ = f.read()
