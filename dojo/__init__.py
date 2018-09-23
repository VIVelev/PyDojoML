import os

from . import (
    base,
    cluster,
    dimred,
    ensemble,
    linear,
    metrics,
    preprocessing,
    svm,
    tree,
    activations,
    exceptions,
)

__all__ = [
    "base",
    "cluster",
    "dimred",
    "ensemble",
    "linear",
    "metrics",
    "preprocessing",
    "svm",
    "tree",
    "activations",
    "exceptions",
]

DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(DIR, "../VERSION"), 'r') as f:
    __version__ = f.read()
