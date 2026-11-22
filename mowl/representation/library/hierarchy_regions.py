"""Regions and spaces of the methods that embed a hierarchy (group ``hierarchy``).

*Boxes with corners.* :class:`~mowl.representation.CornerBox` (in the core) is a box that
also keeps its two corners, so that measures that work with corners (volumes) read them
without the rounding of ``(low + high) / 2 +- (high - low) / 2``. The unit cube is not a
region: it is a constraint of the class model (:class:`~.hierarchy_classes.UnitCubeBoxes`).

*Points on a manifold.* :class:`Point` and its refinements :class:`OrderCone`,
:class:`ConePoint` and :class:`Disk` carry the manifold they live on, so that a measure
needs nothing but the regions. The manifolds are those of :mod:`mowl.nn.order.manifolds`
(the library of the cones, shadow cones and disks); :class:`CurvedBall` adds the curvature
parameter that HiT needs, see its docstring for what merging the two hyperbolic libraries
(:mod:`mowl.nn.order.manifolds` and :mod:`mowl.nn.lm.hyperbolic`) would take.
"""

import math
from dataclasses import dataclass

import torch as th

from mowl.nn.lm import hyperbolic as H
from mowl.nn.order.manifolds import (Euclidean, Hyperboloid, PoincareBall, Sphere,
                                     UpperHalfSpace)
from mowl.representation.regions import CornerBox, Region

__all__ = ["CodedBox", "Point", "OrderCone", "ConePoint", "Disk", "CurvedBall",
           "depth", "MANIFOLDS"]


# -- boxes -------------------------------------------------------------------------------
@dataclass(frozen=True)
class CodedBox(CornerBox):
    """A box with binary codes (BCBox, Zhang et al. 2022): ``code`` ``(B, d)`` is the code of
    a vertex (variant ``vbc``), ``global_code`` ``(d, k)`` the code words shared by all
    vertices (variant ``gbc``, the same for every row)."""

    code: th.Tensor = None
    global_code: th.Tensor = None


# -- points ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Point(Region):
    """A point of a manifold.

    :param centre: ``(B, n)`` coordinates in the model of the manifold.
    :param manifold: The manifold (:mod:`mowl.nn.order.manifolds`).
    :param ids: The row ids the region was looked up with, where it has any: a geometry that
        offers :math:`\\exists r.C` by lookup (CatE) needs the id of the filler.
    """

    centre: th.Tensor
    manifold: object = None
    ids: th.Tensor = None


@dataclass(frozen=True)
class OrderCone(Point):
    """An element of an ordered vector space (order embeddings, CatE); the order is the one of
    :class:`~.hierarchy_measures.OrderViolation`."""


@dataclass(frozen=True)
class ConePoint(Point):
    """A point with the geodesic cone of half-aperture ``aperture`` (``(B,)``) at it
    (hyperbolic entailment cones)."""

    aperture: th.Tensor = None


@dataclass(frozen=True)
class Disk(Point):
    """A point with a radius ``radius`` (``(B,)``, real, possibly negative: a generalised
    formal disk, Suzuki et al. 2019)."""

    radius: th.Tensor = None


# -- spaces ------------------------------------------------------------------------------
class CurvedBall(PoincareBall):
    """The Poincare ball of curvature :math:`-c` (radius :math:`1/\\sqrt{c}`), using the
    functions of :mod:`mowl.nn.lm.hyperbolic` (HiT, OnT).

    **What a merge of the two hyperbolic libraries would take.** ``mowl.nn.order.manifolds``
    is already written on top of ``mowl.nn.lm.hyperbolic`` (it calls ``expmap`` and
    ``distance`` with :math:`c = 1`), so they differ in two things only: the manifolds have
    no curvature parameter and no ``norm`` (the depth of a point, :math:`d(x, 0)`). Merging
    means: (1) give :class:`~mowl.nn.order.manifolds.PoincareBall` a ``c`` and use it in
    ``distance``, ``retract``, ``rgrad`` and ``max_norm``; (2) add ``norm`` to
    :class:`~mowl.nn.order.manifolds.Manifold` (distance to the origin; the origin of the
    half-space is :math:`(0, \\dots, 0, 1)`, that of the hyperboloid :math:`e_0`);
    (3) keep ``mowl.nn.lm.hyperbolic`` as the single home of the formulas. This class is that
    change made in a subclass.
    """

    def __init__(self, c=1.0, **kwargs):
        kwargs.setdefault("max_norm", (1 - 1e-5) / math.sqrt(c))
        super().__init__(**kwargs)
        self.c = float(c)

    def rgrad(self, x, grad):
        return grad * ((1 - self.c * (x * x).sum(-1, keepdim=True)) ** 2 / 4)

    def retract(self, x, v):
        return self.project(H.expmap(x, v, self.c))

    def distance(self, x, y):
        return H.distance(x, y, self.c)

    def norm(self, x):
        """The hyperbolic norm :math:`d_c(x, 0)` (the depth of ``x``)."""
        return H.hyperbolic_norm(x, self.c)


def depth(manifold, x):
    """The distance of ``x`` to the origin of ``manifold`` (its *depth*), ``(...,)``."""
    if hasattr(manifold, "norm"):
        return manifold.norm(x)
    if isinstance(manifold, PoincareBall):
        return manifold.distance(x, th.zeros_like(x))
    if isinstance(manifold, Hyperboloid):
        origin = th.zeros_like(x)
        origin[..., 0] = 1.0
        return manifold.distance(x, origin)
    if isinstance(manifold, UpperHalfSpace):
        origin = th.zeros_like(x)
        origin[..., -1] = 1.0
        return manifold.distance(x, origin)
    raise TypeError(f"{type(manifold).__name__} has no origin, so its points have no depth.")


#: Manifolds by name (the geometries of Disk Embeddings and of the cones).
MANIFOLDS = {"poincare": PoincareBall, "half space": UpperHalfSpace,
             "hyperboloid": Hyperboloid, "sphere": Sphere, "euclidean": Euclidean}
