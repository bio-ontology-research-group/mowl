"""Regions and spaces of the methods that embed a hierarchy (group ``hierarchy``).

*Boxes with corners.* :class:`~mowl.representation.CornerBox` (in the core) is a box that
also keeps its two corners, so that measures that work with corners (volumes) read them
without the rounding of ``(low + high) / 2 +- (high - low) / 2``. The unit cube is not a
region: it is a constraint of the class model (:class:`~.hierarchy_classes.UnitCubeBoxes`).

*Points on a manifold.* :class:`Point` and its refinements :class:`OrderCone`,
:class:`ConePoint` and :class:`Disk` carry the manifold they live on, so that a measure
needs nothing but the regions. The manifolds are those of :mod:`mowl.representation.library.manifolds`
(the library of the cones, shadow cones and disks); :class:`CurvedBall` adds the curvature
parameter that HiT needs, see its docstring for what merging the two hyperbolic libraries
(:mod:`mowl.representation.library.manifolds` and :mod:`mowl.representation.library.poincare_ball`) would take.
"""

import math
from dataclasses import dataclass

import torch as th

from mowl.representation.library import poincare_ball as H
from mowl.representation.library.manifolds import (Euclidean, Hyperboloid, PoincareBall, Sphere,
                                     UpperHalfSpace)
from mowl.representation.regions import ConstructorUnavailable, CornerBox, Region

__all__ = ["meet", "join", "volume", "CodedBox", "Point", "OrderCone", "ConePoint", "Disk", "CurvedBall",
           "depth", "MANIFOLDS"]


# -- boxes -------------------------------------------------------------------------------
# The lattice of boxes (Vilnis et al. 2018, Section 4): the operations on boxes given by their
# corners. They are used by the volume-ratio measure and by the box lattice and Gumbel box
# methods; any method that reads boxes by corners can call them.
def meet(min_a, max_a, min_b, max_b):
    """Meet of two boxes: the intersection box ``(min, max)``. Components with
    ``max < min`` mean that the boxes are disjoint (the meet is the bottom element)."""
    return th.maximum(min_a, min_b), th.minimum(max_a, max_b)


def join(min_a, max_a, min_b, max_b):
    """Join of two boxes: the smallest enclosing box."""
    return th.minimum(min_a, min_b), th.maximum(max_a, max_b)


def volume(mn, mx):
    """Volume (product of side lengths) of a box, 0 if it is empty."""
    return th.relu(mx - mn).prod(dim=-1)


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
    :param manifold: The manifold (:mod:`mowl.representation.library.manifolds`).
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

    def contains(self, points):
        """Whether the points ``(P, n)`` of the Poincare ball lie in the entailment cones,
        ``(B, P)``: the angle at the apex between the axis and the geodesic to the point is
        at most the half-aperture (the cone contains its apex)."""
        from mowl.representation.library.cone_regions import axis_angle
        x = self.centre.unsqueeze(1).expand(-1, points.shape[0], -1)
        y = points.to(self.centre.dtype).unsqueeze(0).expand(len(self), -1, -1)
        angle = axis_angle(x, y)
        apex = th.linalg.norm(x - y, dim=-1) < 1e-6
        return (angle <= self.aperture.unsqueeze(-1) + 1e-6) | apex


@dataclass(frozen=True)
class Disk(Point):
    """A point with a radius ``radius`` (``(B,)``, real, possibly negative: a generalised
    formal disk, Suzuki et al. 2019)."""

    radius: th.Tensor = None

    def contains(self, points):
        """Whether the points ``(P, n)`` lie within ``radius`` of the centre in the metric of
        the manifold, ``(B, P)`` (a disk of negative radius, a generalised disk, contains no
        point: its order is that of :class:`~.hierarchy_measures.Protrusion`, not of sets)."""
        manifold = self.manifold
        x = self.centre.unsqueeze(1).expand(-1, points.shape[0], -1)
        y = points.to(self.centre.dtype).unsqueeze(0).expand(len(self), -1, -1)
        distance = manifold.distance(x, y).reshape(len(self), -1)
        return distance <= self.radius.reshape(-1, 1)


# -- spaces ------------------------------------------------------------------------------
class CurvedBall(PoincareBall):
    """The Poincare ball of curvature :math:`-c` (radius :math:`1/\\sqrt{c}`), using the
    functions of :mod:`mowl.representation.library.poincare_ball` (HiT, OnT).

    **What a merge of the two hyperbolic libraries would take.** ``mowl.representation.library.manifolds``
    is already written on top of ``mowl.representation.library.poincare_ball`` (it calls ``expmap`` and
    ``distance`` with :math:`c = 1`), so they differ in two things only: the manifolds have
    no curvature parameter and no ``norm`` (the depth of a point, :math:`d(x, 0)`). Merging
    means: (1) give :class:`~mowl.representation.library.manifolds.PoincareBall` a ``c`` and use it in
    ``distance``, ``retract``, ``rgrad`` and ``max_norm``; (2) add ``norm`` to
    :class:`~mowl.representation.library.manifolds.Manifold` (distance to the origin; the origin of the
    half-space is :math:`(0, \\dots, 0, 1)`, that of the hyperboloid :math:`e_0`);
    (3) keep ``mowl.representation.library.poincare_ball`` as the single home of the formulas. This class is that
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

    def pairwise_distance(self, x, y, dtype=th.float64):
        """The distances of every row of ``x`` ``(n, d)`` to every row of ``y`` ``(m, d)``,
        ``(n, m)``, computed in ``dtype`` without forming the ``(n, m, d)`` tensor."""
        return H.pairwise_distance(x, y, self.c, dtype=dtype)


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
    raise ConstructorUnavailable(
        f"{type(manifold).__name__} has no origin, so its points have no depth.")


#: Manifolds by name (the geometries of Disk Embeddings and of the cones).
MANIFOLDS = {"poincare": PoincareBall, "half space": UpperHalfSpace,
             "hyperboloid": Hyperboloid, "sphere": Sphere, "euclidean": Euclidean}
