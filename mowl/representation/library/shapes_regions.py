"""Regions of the shapes group: points, octagon products, prisms, ellipsoids, vectors, linked
regions and soft boxes.

Like every region (:mod:`mowl.representation.regions`) they are immutable containers of
tensors with a leading batch dimension, and know the operations their family is closed
under. How two regions are compared is the business of a measure
(:mod:`mowl.representation.library.shapes_measures`).
"""

from dataclasses import dataclass, replace
from typing import Any, Optional

import torch as th

from mowl.nn.el.octagon import geometry as G
from mowl.nn.el.prism.losses import NORMALS, supports
from mowl.nn.el.concept2box.losses import gumbel_intersection
from mowl.representation.regions import Box, Region

__all__ = ["Point", "OctagonProduct", "Prism", "Ellipsoid", "Vector", "Linked", "SoftBox"]


@dataclass(frozen=True)
class Point(Region):
    """A point; the region of an individual where no family of regions is attached to it, or
    an entity of a knowledge graph model, optionally with a *bump* vector (BoxE).

    :param centre: ``(B, d)``.
    :param bump: ``(B, d)`` or ``None``.
    """

    centre: th.Tensor
    bump: th.Tensor = None

    def translate(self, vector):
        return Point(self.centre + vector, self.bump)

    def is_empty(self):
        """Never: a point is a region of no extent, not the empty one."""
        return th.zeros(len(self), dtype=th.bool, device=self.centre.device)


@dataclass(frozen=True)
class OctagonProduct(Region):
    r"""A product of 2-D octagons, one on each consecutive pair of coordinates
    (:mod:`mowl.nn.el.octagon.geometry`). Closed under translation and intersection.

    :param bounds: Support values on the eight facet directions, ``(B, P, 8)``,
        ``P = d / 2``.
    :param centre: ``(B, d)``: a point of the region, where the class model knows one
        (``None`` for results of an intersection).
    """

    bounds: th.Tensor
    centre: Optional[th.Tensor] = None

    def __len__(self):
        return self.bounds.shape[0]

    def is_empty(self):
        """A product is empty iff one of its octagons is (the tightened bounds have a
        negative width: :func:`mowl.nn.el.octagon.geometry.product_emptiness_slack`)."""
        return G.product_emptiness_slack(self.bounds) < 0

    @classmethod
    def of_point(cls, centre):
        """The point ``centre`` (``(B, d)``, ``d`` even) as an octagon product of no
        extent."""
        pairs = centre.reshape(centre.shape[0], -1, 2)
        return cls(G.from_center_radius(pairs, th.zeros(*pairs.shape[:2], 8,
                                                        dtype=centre.dtype,
                                                        device=centre.device)), centre)

    @classmethod
    def of_box(cls, box):
        """The octagon product that equals the box ``box`` (even dimension)."""
        centre = box.centre.reshape(len(box), -1, 2)
        offset = box.offset.reshape(len(box), -1, 2)
        return cls(G.from_box(centre, offset), box.centre)

    def translate(self, vector):
        pairs = vector.reshape(vector.shape[0], -1, 2)
        centre = None if self.centre is None else self.centre + vector
        return OctagonProduct(G.translate(self.bounds, pairs), centre)

    def intersect(self, other):
        bounds = th.minimum(self.bounds, other.bounds)
        widths = G.widths(bounds)
        gap = th.linalg.norm(th.relu(-widths).reshape(widths.shape[0], -1), dim=-1,
                             keepdim=True)
        # Not tightened: the measures tighten on demand, and the tightening of the bounds of
        # an empty intersection is not idempotent.
        return OctagonProduct(bounds), gap


@dataclass(frozen=True)
class Prism(Region):
    r"""A triangle times a box: :math:`T \times B \subset \mathbb{R}^2 \times
    \mathbb{R}^{d-2}` (Federico et al. 2026). Closed under translation and intersection.

    :param supports: Support values of the triangle on its three fixed normals, ``(B, 3)``.
    :param centre: Centre of the box, ``(B, d - 2)``.
    :param offset: Half-widths of the box, ``(B, d - 2)``.
    """

    supports: th.Tensor
    centre: th.Tensor
    offset: th.Tensor

    def is_empty(self):
        """The triangle is empty where its supports sum to less than zero, and the box where
        its upper corner lies below its lower one."""
        return (self.supports.sum(dim=-1) < 0) | (self.offset < 0).any(dim=-1)

    @classmethod
    def of_point(cls, point):
        """The point ``point`` (``(B, d)``) as a prism of no extent."""
        zero = th.zeros(point.shape[0], 1, dtype=point.dtype, device=point.device)
        return cls(supports(point[:, :2], zero), point[:, 2:], th.zeros_like(point[:, 2:]))

    def translate(self, vector):
        normals = NORMALS.to(device=self.supports.device, dtype=self.supports.dtype)
        return Prism(self.supports + vector[:, :2] @ normals.T, self.centre + vector[:, 2:],
                     self.offset)

    def intersect(self, other, signed=False):
        """The intersection and its gap (zero iff it is not empty). ``signed=True``: the
        half-widths of the box part are :math:`(upper - lower) / 2`, negative where the
        boxes are apart (as in the prism module's definitions of classes), instead of
        their absolute value."""
        h = th.minimum(self.supports, other.supports)
        lower = th.maximum(self.centre - self.offset, other.centre - other.offset)
        upper = th.minimum(self.centre + self.offset, other.centre + other.offset)
        triangle_gap = th.relu(-h.sum(dim=-1, keepdim=True))
        box_gap = th.linalg.norm(th.relu(lower - upper), dim=-1, keepdim=True)
        gap = triangle_gap + box_gap
        half = (upper - lower) / 2.0
        return Prism(h, (lower + upper) / 2.0, half if signed else th.abs(half)), gap


@dataclass(frozen=True)
class Ellipsoid(Region):
    r"""The ellipsoid :math:`\{x : \sum_j ((x_j - c_j) / b_j)^2 \le r^2\}`: centre ``c``,
    positive axes ``b`` and a scale ``r``. Closed under translation only (the intersection
    of two ellipsoids is not an ellipsoid).

    :param centre, axes: ``(B, d)``.
    :param radius: The scale :math:`r`, ``(B, 1)``, non-negative.
    """

    centre: th.Tensor
    axes: th.Tensor
    radius: th.Tensor

    def translate(self, vector):
        return replace(self, centre=self.centre + vector)


@dataclass(frozen=True)
class Vector(Region):
    """A vector of a space whose elements mean something (the intensional space of EIKE):
    it has no extent, only a direction and a length.

    :param centre: ``(B, d)``.
    """

    centre: th.Tensor


@dataclass(frozen=True)
class Linked(Region):
    """Two regions of two linked spaces for one name: ``first`` is what it contains,
    ``second`` (or ``None``) what it means.

    A region that is built from it (a translation, an existential restriction) has no
    ``second``: it means nothing of its own.
    """

    first: Any
    second: Any = None

    def __len__(self):
        return len(self.first)

    def translate(self, vector):
        return Linked(self.first.translate(vector), None)


@dataclass(frozen=True)
class SoftBox(Box):
    """A box with Gumbel-smoothed boundaries (:math:`\\beta` is the scale). The offsets of
    an intersection can be negative: the soft intersection of two disjoint boxes is a box
    of negative width, which still has a small positive volume.

    :param beta: Gumbel scale.
    :param ids: Classes the box belongs to, where the box depends on its context (the
        relation transform of Concept2Box needs the class, not only the region).
    """

    beta: float = 1.0
    ids: Optional[th.Tensor] = None

    def translate(self, vector):
        return replace(self, centre=self.centre + vector, ids=None)

    def scale(self, factor):
        return replace(self, centre=self.centre * factor, offset=self.offset * factor.abs(),
                       ids=None)

    def grow(self, offset):
        return replace(self, offset=self.offset + offset, ids=None)

    def intersect(self, other):
        lo, hi = gumbel_intersection(self.lower, self.upper, other.lower, other.upper,
                                     self.beta)
        return SoftBox((lo + hi) / 2, (hi - lo) / 2, self.beta), th.zeros_like(lo[:, :1])
