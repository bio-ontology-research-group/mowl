"""Regions: what a class model returns for a batch of classes.

A region object holds tensors with a leading batch dimension and knows the operations
under which its family is closed (shifting, scaling, intersecting). How two regions are
compared is not its business: that is a measure (:mod:`mowl.representation.measures`).
"""

from dataclasses import dataclass, replace

import torch as th

__all__ = ["Region", "Ball", "Box", "ConstructorUnavailable"]


class ConstructorUnavailable(TypeError):
    """The region family is not closed under the requested constructor (two balls do not
    intersect in a ball), or the geometry does not define it."""


class Region:
    """Base class of regions."""

    def translate(self, vector):
        """The region shifted by ``vector`` (``(B, d)``)."""
        raise ConstructorUnavailable(f"{type(self).__name__} cannot be translated.")

    def scale(self, factor):
        """The region scaled by ``factor`` per coordinate (``(B, d)``)."""
        raise ConstructorUnavailable(f"{type(self).__name__} cannot be scaled per coordinate.")

    def intersect(self, other):
        """``(region, gap)``: the intersection, and a cost ``(B, 1)`` that is zero iff the
        intersection is not empty."""
        raise ConstructorUnavailable(
            f"The intersection of two {type(self).__name__} regions is not a "
            f"{type(self).__name__}.")

    def __len__(self):
        return self.centre.shape[0]


@dataclass(frozen=True)
class Ball(Region):
    """Balls :math:`\\{x : \\lVert x - c \\rVert \\le r\\}`.

    :param centre: ``(B, d)``.
    :param radius: ``(B, 1)``, non-negative.
    :param raw: The stored (signed) radius parameter, where the class model keeps one;
        only objectives that reproduce a published loss read it.
    """

    centre: th.Tensor
    radius: th.Tensor
    raw: th.Tensor = None

    def translate(self, vector):
        return replace(self, centre=self.centre + vector)


@dataclass(frozen=True)
class Box(Region):
    """Axis-parallel boxes :math:`[c - o, c + o]`.

    :param centre: ``(B, d)``.
    :param offset: ``(B, d)``, non-negative half-widths.
    """

    centre: th.Tensor
    offset: th.Tensor

    @property
    def lower(self):
        return self.centre - self.offset

    @property
    def upper(self):
        return self.centre + self.offset

    def translate(self, vector):
        return Box(self.centre + vector, self.offset)

    def scale(self, factor):
        return Box(self.centre * factor, self.offset * factor.abs())

    def grow(self, offset):
        """The Minkowski sum with the box ``[-offset, offset]``."""
        return Box(self.centre, self.offset + offset)

    def intersect(self, other):
        lower = th.maximum(self.lower, other.lower)
        upper = th.minimum(self.upper, other.upper)
        gap = th.linalg.norm(th.relu(lower - upper), dim=1, keepdim=True)
        # |upper - lower| / 2: a box also when the two do not intersect (the gap says so).
        return Box((lower + upper) / 2, th.abs(upper - lower) / 2), gap
