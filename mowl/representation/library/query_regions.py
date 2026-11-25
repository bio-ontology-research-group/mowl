"""Regions, a class model and measures for the query embedding methods (BetaE, ConE, AConE).

* :class:`BetaRegion`: a product of Beta distributions, one per coordinate.
* :class:`SectorCone`: a product of circular sectors, one per coordinate (an axis angle and
  an aperture; the sector covers the angles within ``aperture`` of the axis).
* :class:`ConeClasses`: a class is a sector cone (a class model for
  :class:`~mowl.models.ComposedEL` as well).
* :class:`ConeBounds`: sector cones compared by angle against aperture.
* :class:`BetaKL`: Beta distributions compared by the Kullback-Leibler divergence.

Neither family is closed under intersection, so :meth:`Region.intersect` raises
:class:`~mowl.representation.ConstructorUnavailable` for both: the query methods intersect
with a learned operator (:mod:`mowl.representation.library.query_parts`) that is not the
set intersection.
"""

import math
from dataclasses import dataclass, fields

import torch as th
import torch.nn as nn

from mowl.representation.class_models import Classes
from mowl.representation.measures import RegionMeasure
from mowl.representation.methods import register_class_model, register_measure
from mowl.representation.regions import Region, ConstructorUnavailable

pi = math.pi

__all__ = ["BetaRegion", "SectorCone", "ConeClasses", "ConeBounds", "BetaKL", "as_state",
           "from_state", "lift", "wrap_angle", "to_axis", "to_aperture"]


def to_axis(x):
    """Raw parameter to an axis angle in ``(-pi, pi)``."""
    return th.tanh(x) * pi


def to_aperture(x):
    """Raw parameter to an aperture in ``(0, pi)``."""
    return th.tanh(2 * x) * pi / 2 + pi / 2


def wrap_angle(angle):
    """The angle in ``(-pi, pi]`` of the unit complex number ``exp(i * angle)``."""
    return th.atan2(th.sin(angle), th.cos(angle))


@dataclass(frozen=True)
class BetaRegion(Region):
    """Beta distributions per coordinate (BetaE): ``alpha``, ``beta`` of shape ``(B, d)``,
    positive. The "region" is the density; its mean is :attr:`centre`."""

    alpha: th.Tensor
    beta: th.Tensor

    @property
    def centre(self):
        return self.alpha / (self.alpha + self.beta)

    def invert(self):
        """The negation of BetaE: parameters inverted, so that the density is high where
        the original is low."""
        return BetaRegion(1.0 / self.alpha, 1.0 / self.beta)


@dataclass(frozen=True)
class SectorCone(Region):
    """Circular sectors per coordinate (ConE): ``axis`` in ``[-pi, pi]`` and ``aperture`` in
    ``[0, pi]``, both ``(B, d)``. AConE's published rotation wraps the aperture to
    ``(-pi, pi]``, so that it can be negative (its absolute value is what counts)."""

    axis: th.Tensor
    aperture: th.Tensor

    @property
    def centre(self):
        return self.axis

    def rotate(self, d_axis, d_aperture, aperture="clip"):
        """Rotation: the angles are added; the axis is wrapped. ``aperture="clip"``
        (default) keeps the aperture in ``[0, pi]``; ``"wrap"`` wraps it to ``(-pi, pi]`` as
        the published AConE does (a negative aperture)."""
        if aperture not in ("clip", "wrap"):
            raise ValueError("aperture must be 'clip' or 'wrap'.")
        new = self.aperture + d_aperture
        new = th.clamp(new, 0.0, pi) if aperture == "clip" else wrap_angle(new)
        return SectorCone(wrap_angle(self.axis + d_axis), new)

    def contains(self, points):
        """Whether the points ``(P, d)`` (one angle per coordinate) lie in every sector,
        ``(B, P)``: within ``|aperture|`` of the axis on the circle."""
        turn = points.unsqueeze(0) - self.axis.unsqueeze(1)
        gap = th.abs(th.atan2(th.sin(turn), th.cos(turn)))
        return (gap <= th.abs(self.aperture).unsqueeze(1)).all(dim=-1)

    def complement(self, signed=False):
        """The closure of the complement *per coordinate*: the axis turned by ``pi``, the
        aperture ``pi - |aperture|`` (default, in ``[0, pi]``). ``signed=True``: ``pi -
        aperture`` as published (ConE), which exceeds ``pi`` for a negative aperture.

        A cone is a product of sectors, and the product of the complements is not the
        complement of the product (it misses the points that are outside in some
        coordinates and inside in others): this negation is approximate, as ConE's."""
        axis = th.where(self.axis >= 0, self.axis - pi, self.axis + pi)
        return SectorCone(axis, pi - (self.aperture if signed else th.abs(self.aperture)))


def as_state(region):
    """The tuple of tensors of a region (the *state* of the query models)."""
    return tuple(getattr(region, f.name) for f in fields(region))


def from_state(region_class, state):
    """The region of the family ``region_class`` with the tensors ``state``."""
    return region_class(*state)


def lift(region):
    """The region with an inserted dimension 1 (``(B, d)`` to ``(B, 1, d)``)."""
    return type(region)(*(t.unsqueeze(1) for t in as_state(region)))


class ConeClasses(Classes):
    """A class is a sector cone per coordinate: an axis ``tanh(x) * pi`` and an aperture
    ``tanh(2 y) * pi / 2 + pi / 2`` of stored parameters ``x`` and ``y``.

    :param centres, apertures: Existing tables (of another module, which then shares its
        parameters).

    Part (vocabulary value): ``sector cone``. Used by: no method by default (``acone`` and
    ``cone_qe`` read sector cones through their query models).
    """

    region = SectorCone
    part = "sector cone"
    exact_conjunction = False

    @property
    def extra_parts(self):
        """The complement is taken per coordinate (:meth:`SectorCone.complement`): the
        product of the complements is a strict subset of the complement of the product, so
        negation is approximate (it under-approximates :math:`\\neg C`)."""
        return {"negation": "approximate"}

    def __init__(self, size=None, embed_dim=None, centres=None, apertures=None):
        super().__init__()
        if centres is None:
            centres = nn.Embedding(size, embed_dim)
            nn.init.uniform_(centres.weight, a=-1, b=1)
        if apertures is None:
            apertures = nn.Embedding(centres.num_embeddings, centres.embedding_dim)
            nn.init.uniform_(apertures.weight, a=-0.5, b=0.5)
        self.centres, self.apertures = centres, apertures

    def forward(self, ids):
        return SectorCone(to_axis(self.centres(ids)), to_aperture(self.apertures(ids)))


def _arc_distance(a, b):
    """Angular distance in ``[0, pi]``."""
    return th.abs(wrap_angle(a - b))


class ConeBounds(RegionMeasure):
    """Sector cones compared per coordinate by angle against aperture.

    With ``d`` the angular distance of the axes and ``alpha`` the (absolute) apertures,
    :math:`A \\subseteq B` iff :math:`\\min(d + \\alpha_A, \\pi) \\le \\alpha_B` in every
    coordinate, the sectors overlap iff :math:`d \\le \\alpha_A + \\alpha_B`. As for boxes
    (:class:`~mowl.representation.Bounds`) the cost is the norm of the violations, and two
    cones are apart if they are separated in at least one coordinate.

    Part (vocabulary value): ``angle``. Used by: ``acone``, ``cone_qe``.
    """

    accepts = (SectorCone,)
    part = "angle"
    exact_intersection = False

    @staticmethod
    def _norm(x):
        return th.linalg.norm(th.relu(x), dim=1, keepdim=True)

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        reach = th.clamp(_arc_distance(a.axis, b.axis) + th.abs(a.aperture), max=pi)
        return self._norm(reach - th.abs(b.aperture) + margin)

    def meets(self, a, b, margin=0.0):
        self.check(a, b)
        return self._norm(_arc_distance(a.axis, b.axis) - th.abs(a.aperture)
                          - th.abs(b.aperture) + margin)

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        overlap = (th.abs(a.aperture) + th.abs(b.aperture)
                   - _arc_distance(a.axis, b.axis) + margin)
        return th.relu(overlap.min(dim=1, keepdim=True).values)

    def empty(self, a, margin=0.0):
        """Zero iff some sector has no aperture (the product then contains no sampled
        point; the norm of the apertures, used before, was zero only if all had none)."""
        self.check(a)
        return th.relu(th.abs(a.aperture).amin(dim=1, keepdim=True) + margin)

    def nonempty(self, a, epsilon=0.0):
        """Zero iff every sector has aperture at least ``epsilon`` (the norm, used before,
        was zero also for a product with a sector of no aperture)."""
        self.check(a)
        return th.relu(epsilon - th.abs(a.aperture).amin(dim=1, keepdim=True))


def beta_kl(a1, b1, a2, b2):
    """``KL(Beta(a1, b1) || Beta(a2, b2))``, elementwise."""
    def lbeta(a, b):
        return th.lgamma(a) + th.lgamma(b) - th.lgamma(a + b)
    return (lbeta(a2, b2) - lbeta(a1, b1) + (a1 - a2) * th.digamma(a1)
            + (b1 - b2) * th.digamma(b1) + (a2 - a1 + b2 - b1) * th.digamma(a1 + b1))


class BetaKL(RegionMeasure):
    """Beta distributions compared by :math:`KL(A \\Vert B)` summed over the coordinates.

    This is the divergence BetaE scores an answer with (the entity's distribution against
    the query's). It is zero iff the two distributions are equal, which is a *membership*
    cost, not an inclusion: two Beta distributions have no containment order. Only
    :meth:`inside` is defined; ``margin`` is added before the hinge.

    Part (vocabulary value): ``divergence``. Used by: ``betae``.
    """

    accepts = (BetaRegion,)
    part = "divergence"
    exact_intersection = False
    reading = "score"

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        kl = beta_kl(a.alpha, a.beta, b.alpha, b.beta).sum(dim=-1, keepdim=True)
        return th.relu(kl + margin)

    def _undefined(self, *args, **kwargs):
        raise ConstructorUnavailable(
            "Beta distributions have no notion of disjointness, overlap or emptiness; "
            "only the divergence (inside) is defined.")

    meets = apart = empty = nonempty = _undefined


register_class_model("sector cone")(ConeClasses)
register_measure("sector angle", default_for=(SectorCone,))(ConeBounds)
register_measure("divergence", default_for=(BetaRegion,))(BetaKL)
