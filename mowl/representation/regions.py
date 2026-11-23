"""Regions: what a class model returns for a batch of classes.

A region object holds tensors with a leading batch dimension and knows the operations
under which its family is closed (shifting, scaling, intersecting). How two regions are
compared is not its business: that is a measure (:mod:`mowl.representation.measures`).
"""

from dataclasses import dataclass, replace

import torch as th

import math

__all__ = ["Region", "Ball", "BallLens", "Box", "CornerBox", "ConstructorUnavailable",
           "ConstructorNotImplemented", "MarginUnavailable", "MarginNotImplemented",
           "family_refusal"]


class ConstructorUnavailable(TypeError):
    """The combination of parts is refused at build time because it is *undefined*: the
    region family is not closed under the requested constructor, the quantity does not exist
    for the family, or the parts live in different spaces. (A combination that is defined but
    whose formula is not written is the subclass :class:`ConstructorNotImplemented`.)"""

    def reworded(self, before="", after=""):
        """A refusal of the same kind (undefined or not implemented, the same class) whose
        message is ``before + (this message without any "not implemented: ") + after``; for
        code that adds where a refusal was raised without turning one kind into the other."""
        message = str(self)
        if isinstance(self, ConstructorNotImplemented):
            message = message[len(ConstructorNotImplemented.PREFIX):]
        return type(self)(f"{before}{message}{after}")


class ConstructorNotImplemented(ConstructorUnavailable):
    """The combination is mathematically defined (or may well be), but the formula is not
    written in mOWL: the message starts with ``"not implemented: "``. Every other
    :class:`ConstructorUnavailable` says that the combination is undefined (a space that
    has no such region, a quantity that does not exist for the family, a loss that no
    constant can change). The documentation of the swaps counts the two apart."""

    PREFIX = "not implemented: "

    def __init__(self, message=""):
        message = str(message)
        super().__init__(message if message.startswith(self.PREFIX) else self.PREFIX + message)


class MarginUnavailable(ConstructorUnavailable, ValueError):
    """A measure (or objective) that cannot read the margin it is given: a graded truth
    value, a signed score that a constant shift does not change, a log-probability under a
    slack. A :class:`ValueError` as well, for the code that checks margins as values."""


class MarginNotImplemented(ConstructorNotImplemented, MarginUnavailable):
    """A margin that has a meaning for the measure but whose reading is not written (a slack
    as the shrunk region)."""


def family_refusal(subject, accepted, region):
    """The refusal of ``subject`` (a measure, a relation model: a name) for a ``region`` of a
    family it was not written for, ``accepted`` being the families it was. Both in
    :math:`\\mathbb{R}^d`: the comparison is defined and only its formula is not written
    (:class:`ConstructorNotImplemented`); another space: undefined, said so."""
    names = [c.__name__ for c in accepted]
    given = region.__name__ if isinstance(region, type) else type(region).__name__
    if getattr(region, "space", None) == "euclidean" and any(
            getattr(c, "space", None) == "euclidean" for c in accepted):
        return ConstructorNotImplemented(
            f"{subject} has formulas for {names} regions only; a {given} is a region of the "
            f"same space (R^d), so the comparison is defined, but its formula is not written.")
    return ConstructorUnavailable(
        f"{subject} compares {names} regions, which live in another space than a {given}: "
        f"there is no comparison of the two.")


class Region:
    """Base class of regions."""

    #: The space the regions are subsets of: ``"euclidean"`` for the regions of
    #: :math:`\\mathbb{R}^d` (balls, boxes, octagons, prisms, ellipsoids, unions of boxes),
    #: ``None`` for the others (points of a manifold, cones, orders, fuzzy sets, element
    #: sets, symbols). Two families with the same space can be compared by a formula that is
    #: not written yet; two with different spaces have no comparison.
    space = None

    def _refuse(self, what):
        """The refusal of an operation a region of this family does not have: for a region
        of :math:`\\mathbb{R}^d` the operation exists (a shift, a scaling, an enlargement, a
        contraction are defined for every subset), so only its formula is missing
        (:class:`ConstructorNotImplemented`); for another space it is undefined."""
        if self.space == "euclidean":
            return ConstructorNotImplemented(f"{type(self).__name__} regions {what} "
                                             f"(no formula is written for the family).")
        return ConstructorUnavailable(f"{type(self).__name__} regions cannot be {what}.")

    def translate(self, vector):
        """The region shifted by ``vector`` (``(B, d)``)."""
        raise self._refuse("translated")

    def scale(self, factor):
        """The region scaled by ``factor`` per coordinate (``(B, d)``)."""
        raise self._refuse("scaled per coordinate")

    def rotate(self, d_axis, d_aperture, aperture="clip"):
        """The region turned by the angles ``d_axis``, its aperture changed by
        ``d_aperture`` (families of angular cones)."""
        raise ConstructorUnavailable(f"{type(self).__name__} cannot be rotated.")

    def intersect(self, other):
        """``(region, gap)``: the intersection, and a cost ``(B, 1)`` that is zero iff the
        intersection is not empty."""
        raise ConstructorUnavailable(
            f"The intersection of two {type(self).__name__} regions is not a "
            f"{type(self).__name__}.")

    def grow(self, amount):
        """The region enlarged by ``amount`` in every direction (the Minkowski sum with a
        ball or box of that size)."""
        raise self._refuse("enlarged")

    def complement(self):
        """The region of :math:`\\neg C` (families closed under complement: fuzzy sets,
        cones)."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no complement.")

    def union(self, other):
        """The region of :math:`C \\sqcup D` (families closed under union: fuzzy sets,
        unions of boxes)."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no union.")

    #: Region classes this family is an exact special case of (a point is a box of no
    #: width): a measure of those families compares it through :meth:`view_as`.
    views = ()

    @classmethod
    def viewable_as(cls, families):
        """Whether the regions of this class are exactly regions of one of ``families``
        (declared by this class, :attr:`views`, or by the family, :meth:`represents`)."""
        families = families if isinstance(families, tuple) else (families,)
        return any(issubclass(view, families) for view in cls.views) or any(
            isinstance(f, type) and issubclass(f, Region) and f.represents(cls)
            for f in families)

    def view_as(self, families):
        """The same set as a region of one of ``families`` (``None`` if it is none)."""
        families = families if isinstance(families, tuple) else (families,)
        for family in families:
            if isinstance(family, type) and issubclass(family, Region) \
                    and family.represents(type(self)):
                return family.from_region(self)
        return None

    @classmethod
    def represents(cls, region_class):
        """Whether every region of ``region_class`` is exactly a region of this family (an
        octagon product represents boxes); default: none other."""
        return False

    @classmethod
    def from_region(cls, region):
        """The region of this family that equals ``region`` (see :meth:`represents`)."""
        raise ConstructorUnavailable(f"{cls.__name__} does not represent "
                                     f"{type(region).__name__} regions.")

    def volume(self):
        """``(B, 1)``: the Lebesgue volume of each region (zero for an empty one). Families
        with a closed form have it; a region whose volume is not written refuses."""
        raise ConstructorNotImplemented(
            f"the volume of {type(self).__name__} regions (no formula is written for it).")

    def contract(self):
        """The region contracted to its centre: the degenerate region of the same family
        (a box or ball of size zero) at the centre."""
        raise self._refuse("contracted to their centre")

    def detach_size(self):
        """The same region with its size (radius, half-widths) detached from the graph, so
        that a loss computed from it moves only the location (the frozen negatives of the
        box lattice model)."""
        raise self._refuse("detached from their size")

    def is_empty(self):
        """``(B,)`` boolean tensor: whether the region of each row is the empty set. A
        family that can say so overrides this; for the others it is an error rather than a
        guess (a fuzzy set or a cone is not empty because it has no tensor of zeros)."""
        raise ConstructorUnavailable(f"{type(self).__name__} cannot say whether it is empty.")

    def contains(self, points):
        """``(B, P)`` boolean tensor: whether each of the points ``(P, d)`` lies in the
        region of each row (the closed set). This is the meaning of the region that the
        measures are tested against (``tests/representation/test_soundness_*``). Only a
        region that *is* a set of points of a space has it; a fuzzy set, a distribution or
        an element of a lattice has another meaning and no point-membership test."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} is not a set of points of a space (it has no "
            f"point-membership test).")

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

    space = "euclidean"

    def translate(self, vector):
        return replace(self, centre=self.centre + vector)

    def contract(self):
        return replace(self, radius=th.zeros_like(self.radius), raw=None)

    def detach_size(self):
        return replace(self, radius=self.radius.detach(),
                       raw=None if self.raw is None else self.raw.detach())

    def is_empty(self):
        """Never: the radius is non-negative (a ball of radius 0 is a point)."""
        return th.zeros(len(self), dtype=th.bool, device=self.centre.device)

    def contains(self, points):
        """Whether the points ``(P, d)`` lie in the closed balls, ``(B, P)``."""
        diff = points.to(self.centre.dtype).unsqueeze(0) - self.centre.unsqueeze(1)
        return th.linalg.norm(diff, dim=-1) <= self.radius

    def grow(self, amount):
        """The ball enlarged by ``amount`` (``(B, 1)``) in every direction (the Minkowski
        sum with a ball of that radius); the signed ``raw`` radius is dropped."""
        return replace(self, radius=self.radius + amount, raw=None)

    def volume(self):
        """``(B, 1)``: :math:`\\pi^{d/2} r^d / \\Gamma(d/2 + 1)`."""
        d = self.centre.shape[-1]
        return math.exp(_log_unit_ball(d)) * self.radius ** d

    def intersect(self, other):
        """``(lens, gap)``: the intersection of two balls is not a ball but a lens
        (:class:`BallLens`, which has a volume); ``gap`` is zero iff the balls meet."""
        if not isinstance(other, Ball):
            raise ConstructorUnavailable(
                f"The intersection of a Ball with {type(other).__name__} is not a region "
                f"of one of the two families.")
        distance = th.linalg.norm(self.centre - other.centre, dim=-1, keepdim=True)
        return BallLens(self, other), th.relu(distance - self.radius - other.radius)


def _log_unit_ball(d):
    """:math:`\\log` of the volume of the unit ball of dimension ``d``."""
    return d / 2 * math.log(math.pi) - math.lgamma(d / 2 + 1)


_QUADRATURE = {}


def _sin_power_integral(angle, power, nodes=64):
    """:math:`\\int_0^{\\theta} \\sin^n \\varphi\\, d\\varphi` by Gauss-Legendre
    quadrature (differentiable in ``angle``)."""
    import numpy as np
    if nodes not in _QUADRATURE:
        _QUADRATURE[nodes] = np.polynomial.legendre.leggauss(nodes)
    x, w = _QUADRATURE[nodes]
    x = th.as_tensor(x, dtype=angle.dtype, device=angle.device)
    w = th.as_tensor(w, dtype=angle.dtype, device=angle.device)
    phi = angle.unsqueeze(-1) / 2 * (x + 1)
    return (th.sin(phi) ** power * w).sum(dim=-1) * angle / 2


@dataclass(frozen=True)
class BallLens(Region):
    """The intersection of two balls (a lens); it has a volume and a membership test, and is
    not a ball. ``first`` and ``second`` are :class:`Ball` regions of the same batch."""

    first: Ball
    second: Ball

    space = "euclidean"

    def __len__(self):
        return self.first.centre.shape[0]

    def contains(self, points):
        return self.first.contains(points) & self.second.contains(points)

    def volume(self):
        """``(B, 1)``: the two caps cut by the radical plane,
        :math:`V_{d-1}\\,(r_1^d I_d(\\theta_1) + r_2^d I_d(\\theta_2))` with
        :math:`I_d(\\theta) = \\int_0^\\theta \\sin^d`, :math:`\\cos\\theta_i = a_i / r_i`
        and :math:`a_i` the distance of the plane from the centre ``i`` (clamped, so that
        disjoint balls give 0 and nested balls the smaller one)."""
        a, b = self.first, self.second
        d = a.centre.shape[-1]
        dist = th.linalg.norm(a.centre - b.centre, dim=-1, keepdim=True).clamp_min(1e-12)
        r1, r2 = a.radius.clamp_min(1e-12), b.radius.clamp_min(1e-12)
        a1 = (dist ** 2 + r1 ** 2 - r2 ** 2) / (2 * dist)
        a2 = dist - a1
        theta1 = th.acos((a1 / r1).clamp(-1.0, 1.0))
        theta2 = th.acos((a2 / r2).clamp(-1.0, 1.0))
        cap = lambda r, t: r ** d * _sin_power_integral(t, d)      # noqa: E731
        total = math.exp(_log_unit_ball(d - 1)) * (cap(r1, theta1) + cap(r2, theta2))
        return th.where(dist >= a.radius + b.radius, th.zeros_like(total), total)


@dataclass(frozen=True)
class Box(Region):
    """Axis-parallel boxes :math:`[c - o, c + o]`.

    :param centre: ``(B, d)``.
    :param offset: ``(B, d)``, non-negative half-widths.
    """

    centre: th.Tensor
    offset: th.Tensor

    space = "euclidean"

    @property
    def lower(self):
        return self.centre - self.offset

    @property
    def upper(self):
        return self.centre + self.offset

    def volume(self):
        """The product of the side lengths, zero where the box is empty."""
        return th.relu(self.upper - self.lower).prod(dim=-1, keepdim=True)

    def contract(self):
        return replace(self, offset=th.zeros_like(self.offset))

    def detach_size(self):
        return replace(self, offset=self.offset.detach())

    def is_empty(self):
        """Whether the upper corner lies below the lower one in some coordinate. With
        non-negative half-widths that never happens; a box of negative half-width is the
        empty box (the fixed bottom class of ELK-Em, a signed BoxEL scale, the signed
        intersection of two apart boxes)."""
        return (self.upper < self.lower).any(dim=-1)

    def contains(self, points):
        """Whether the points ``(P, d)`` lie in the closed boxes ``[lower, upper]``,
        ``(B, P)``; a box whose upper corner lies below its lower one in some coordinate
        contains none."""
        x = points.to(self.centre.dtype).unsqueeze(0)
        inside = (x >= self.lower.unsqueeze(1)) & (x <= self.upper.unsqueeze(1))
        return inside.all(dim=-1)

    def translate(self, vector):
        return Box(self.centre + vector, self.offset)

    def scale(self, factor):
        return Box(self.centre * factor, self.offset * factor.abs())

    def grow(self, offset):
        """The Minkowski sum with the box ``[-offset, offset]``."""
        return Box(self.centre, self.offset + offset)

    def intersect(self, other, signed=True):
        """The intersection and its ``gap`` (zero iff it is not empty).

        :param signed: ``True`` (default): the half-widths are :math:`(upper - lower) / 2`,
            negative in the coordinates in which the boxes are apart, so that the result is
            exactly the set of points in both (the empty box where they are apart), keeps
            track of how far apart they are (volume measures price it) and intersecting
            again is associative. ``False``: they are :math:`|upper - lower| / 2`, a box that
            is *not* empty also where the two do not intersect (the reading of ELBE's
            conjunction loss, which adds the gap; it was the default before).
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

    def contract(self):
        middle = (self.low + self.high) / 2
        return replace(self, centre=middle, offset=th.zeros_like(self.offset), low=middle,
                       high=middle)

    def detach_size(self):
        middle, half = (self.low + self.high) / 2, ((self.high - self.low) / 2).detach()
        return replace(self, centre=middle, offset=half.abs(), low=middle - half,
                       high=middle + half)

    def translate(self, vector):
        return CornerBox.from_corners(self.low + vector, self.high + vector)

    def scale(self, factor):
        a, b = self.low * factor, self.high * factor
        return CornerBox.from_corners(th.minimum(a, b), th.maximum(a, b))

    def grow(self, offset):
        return CornerBox.from_corners(self.low - offset, self.high + offset)

    def intersect(self, other, signed=True):
        low = th.maximum(self.lower, other.lower)
        high = th.minimum(self.upper, other.upper)
        gap = th.linalg.norm(th.relu(low - high), dim=1, keepdim=True)
        if signed:     # negative half-widths where the boxes are apart, as Box.intersect
            return CornerBox((low + high) / 2, (high - low) / 2, low, high), gap
        # the corners stay exact (an empty box keeps high < low); only the half-widths are
        # read as their absolute value
        return CornerBox.from_corners(low, high), gap
