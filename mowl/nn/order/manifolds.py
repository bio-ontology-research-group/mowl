"""Moved: the manifolds are general parts, now in
:mod:`mowl.representation.library.manifolds`. This module keeps the old import path."""
from mowl.representation.library.manifolds import *  # noqa: F401,F403
from mowl.representation.library.manifolds import (  # noqa: F401
    Euclidean, Hyperboloid, Manifold, PoincareBall, RiemannianSGD, Sphere, UpperHalfSpace)
