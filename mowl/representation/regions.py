"""Regions: what a class model returns for a batch of classes.

A region object holds tensors with a leading batch dimension and knows the operations
under which its family is closed (shifting, scaling, intersecting). How two regions are
compared is not its business: that is a measure (:mod:`mowl.representation.measures`).
"""

from dataclasses import dataclass, replace

import torch as th

__all__ = ["Region", "Ball", "Box", "CornerBox", "ConstructorUnavailable"]


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

    def grow(self, amount):
        """The region enlarged by ``amount`` in every direction (the Minkowski sum with a
        ball or box of that size)."""
        raise ConstructorUnavailable(f"{type(self).__name__} cannot be enlarged.")

    def complement(self):
        """The region of :math:`\\neg C` (families closed under complement: fuzzy sets,
        cones)."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no complement.")

    def union(self, other):
        """The region of :math:`C \\sqcup D` (families closed under union: fuzzy sets,
        unions of boxes)."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no union.")

    def is_empty(self):
        """``(B,)`` boolean tensor: whether the region of each row is the empty set. A
        family that can say so overrides this; for the others it is an error rather than a
        guess (a fuzzy set or a cone is not empty because it has no tensor of zeros)."""
        raise ConstructorUnavailable(f"{type(self).__name__} cannot say whether it is empty.")

    def __len__(self):
        """The batch size: the leading dimension of the first tensor of the region (not
        every region has a ``centre``)."""
        for value in vars(self).values():
            if isinstance(value, th.Tensor):
                return value.shape[0]
        raise TypeError(f"{type(self).__name__} holds no tensor.")


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

    def is_empty(self):
        """Never: the radius is non-negative (a ball of radius 0 is a point)."""
        return th.zeros(len(self), dtype=th.bool, device=self.centre.device)

    def grow(self, amount):
        """The ball enlarged by ``amount`` (``(B, 1)``) in every direction (the Minkowski
        sum with a ball of that radius); the signed ``raw`` radius is dropped."""
        return replace(self, radius=self.radius + amount, raw=None)


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

    def is_empty(self):
        """Whether the upper corner lies below the lower one in some coordinate. With
        non-negative half-widths that never happens; a box of negative half-width is the
        empty box (the fixed bottom class of ELK-Em, a signed BoxEL scale, the signed
        intersection of two apart boxes)."""
        return (self.upper < self.lower).any(dim=-1)

    def translate(self, vector):
        return Box(self.centre + vector, self.offset)

    def scale(self, factor):
        return Box(self.centre * factor, self.offset * factor.abs())

    def grow(self, offset):
        """The Minkowski sum with the box ``[-offset, offset]``."""
        return Box(self.centre, self.offset + offset)

    def intersect(self, other, signed=False):
        """The intersection and its ``gap`` (zero iff it is not empty).

        :param signed: ``False`` (default): the half-widths of the result are
            :math:`|upper - lower| / 2`, a box also where the two do not intersect.
            ``True``: they are :math:`(upper - lower) / 2`, negative in the coordinates in
            which the boxes are apart, which keeps track of how far apart they are (volume
            measures price it) and makes intersecting again associative.
        """
        lower = th.maximum(self.lower, other.lower)
        upper = th.minimum(self.upper, other.upper)
        gap = th.linalg.norm(th.relu(lower - upper), dim=1, keepdim=True)
        half = (upper - lower) / 2
        return Box((lower + upper) / 2, half if signed else th.abs(half)), gap


@dataclass(frozen=True)
class CornerBox(Box):
    """A box that keeps its corners, so that what reads corners (volumes, the exact-model
    check) gets the numbers the parameters give without the rounding of
    ``(low + high) / 2 +- (high - low) / 2``.

    :param low, high: ``(B, d)``. ``high < low`` in a coordinate is allowed and means an
        empty box (the result of intersecting two disjoint boxes); volumes treat it as 0.
    """

    low: th.Tensor = None
    high: th.Tensor = None

    @classmethod
    def from_corners(cls, low, high, **extra):
        return cls((low + high) / 2, (high - low).abs() / 2, low, high, **extra)

    @property
    def lower(self):
        return self.low

    @property
    def upper(self):
        return self.high

    def translate(self, vector):
        return CornerBox.from_corners(self.low + vector, self.high + vector)

    def scale(self, factor):
        a, b = self.low * factor, self.high * factor
        return CornerBox.from_corners(th.minimum(a, b), th.maximum(a, b))

    def grow(self, offset):
        return CornerBox.from_corners(self.low - offset, self.high + offset)

    def intersect(self, other, signed=False):
        low = th.maximum(self.lower, other.lower)
        high = th.minimum(self.upper, other.upper)
        gap = th.linalg.norm(th.relu(low - high), dim=1, keepdim=True)
        if signed:     # negative half-widths where the boxes are apart, as Box.intersect
            return CornerBox((low + high) / 2, (high - low) / 2, low, high), gap
        return CornerBox.from_corners(low, high), gap
