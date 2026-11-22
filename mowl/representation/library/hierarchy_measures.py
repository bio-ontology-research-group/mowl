"""Measures of the methods that embed a hierarchy (group ``hierarchy``).

====================================  ==================  =================================
measure                               vocabulary entry    used by
====================================  ==================  =================================
:class:`OrderViolation`               order violation     order embeddings, CatE
:class:`LogVolumeRatio`               volume ratio,       box lattice, Gumbel boxes, BCBox
                                      smoothed volume
:class:`ConeAngle`                    angle               hyperbolic entailment cones
:class:`ShadowCone`                   angle               shadow cones
:class:`Protrusion`                   protrusion          disk embeddings
:class:`DistanceDepth`                distance and depth  HiT, OnT
====================================  ==================  =================================

A measure of this group has, besides the ``inside`` of every measure, a ``negative``: the cost
of a pair that is *not* ordered, which is what the published losses of these methods differ
in most (a hinge, a log-probability, the plain energy). ``inside(a, b)`` is the cost of
:math:`a \\sqsubseteq b`; ``a`` is the child and ``b`` the parent.

The energies of the measures that compare against a threshold (cones, order violation,
protrusion, shadow cones) are available in two readings, as for every part of this library:
the default is a cost that is zero exactly where the relation holds and accepts a strict
margin; ``signed=True`` returns the raw signed energy the published code uses (negative
inside the cone or disk) and takes no margin.
"""

import math

import torch as th
import torch.nn.functional as F

from mowl.representation.library.manifolds import Euclidean, UpperHalfSpace
from mowl.representation.measures import Bounds, RegionMeasure
from mowl.representation.methods import DEFAULT_MEASURE, register_measure
from mowl.representation.regions import Ball, Box, ConstructorUnavailable, CornerBox
from mowl.representation.library.hierarchy_regions import (
    CodedBox, ConePoint, Disk, OrderCone, ManifoldPoint, depth, join, meet, volume)
from mowl.representation.library.cone_regions import (
    FORMULATIONS, OriginLight, Shadow, light_for)

__all__ = ["HierarchyMeasure", "EnergyMeasure", "OrderViolation", "LogVolumeRatio",
           "ConeAngle", "ShadowCone", "Protrusion", "DistanceDepth",
           "neg_log1mexp"]

EULER_GAMMA = 0.5772156649015329


class _LogProbability:
    """Mixin of the measures whose cost is a log-probability and has no margin."""

    @staticmethod
    def _no_margin(margin, what):
        if margin:
            raise ValueError(f"{what} is a log-probability and has no margin; pass 0.")


class HierarchyMeasure(_LogProbability, RegionMeasure):
    """Base class: ``inside`` and ``negative``; the other relations of a measure are
    unavailable unless a subclass defines them."""

    exact_intersection = False

    def negative(self, a, b, margin=0.0):
        """The cost of the pair ``a``, ``b`` when it is *not* ordered (low when ``a`` is far
        from lying inside ``b``)."""
        raise NotImplementedError

    def _unavailable(self, what):
        raise ConstructorUnavailable(f"{type(self).__name__} has no {what}.")

    def meets(self, a, b, margin=0.0):
        self._unavailable("overlap")

    def apart(self, a, b, margin=0.0):
        self._unavailable("disjointness")

    def empty(self, a, margin=0.0):
        self._unavailable("empty region")

    def nonempty(self, a, epsilon=0.0):
        self._unavailable("empty region")


class EnergyMeasure(HierarchyMeasure):
    """A measure with a signed energy :math:`e(a, b)` that is :math:`\\le 0` where
    :math:`a` lies inside :math:`b`. Subclasses define :meth:`raw` (``(B, 1)``).

    :param signed: ``inside`` returns the raw energy (published code of shadow cones and disk
        embeddings) instead of the hinge :math:`\\max(0, e + \\gamma)`.
    :param negative_form: ``"hinge"`` (default): :math:`\\max(0, \\gamma - e)`;
        ``"energy"``: the energy itself (shadow cones: the loss of negatives is left to the
        training regime).
    """

    #: Whether the published energy is already hinged at 0 (cones, order violation).
    floored = False

    def __init__(self, signed=False, negative_form="hinge"):
        if negative_form not in ("hinge", "energy"):
            raise ValueError("negative_form must be 'hinge' or 'energy'.")
        if signed and self.floored:
            raise ValueError(f"{type(self).__name__} has a non-negative energy already.")
        self.signed = bool(signed)
        self.negative_form = negative_form

    def raw(self, a, b):
        raise NotImplementedError

    def energy(self, a, b):
        """The energy as published: hinged at 0 where the measure floors, otherwise signed."""
        self.check(a, b)
        e = self.raw(a, b)
        return th.relu(e) if self.floored else e

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        if self.signed:
            if margin:
                raise ValueError("A signed energy has no margin; pass 0.")
            return self.raw(a, b)
        return th.relu(self.raw(a, b) + margin)

    def negative_of(self, energy, margin=0.0):
        """The cost of a pair that is not ordered, from its ``energy``: the hinge
        :math:`\\max(0, \\gamma - e)`, or the energy itself for ``negative_form="energy"``."""
        return energy if self.negative_form == "energy" else th.relu(margin - energy)

    def negative(self, a, b, margin=0.0):
        return self.negative_of(self.energy(a, b), margin)


# -- order violation ----------------------------------------------------------------------
class OrderViolation(EnergyMeasure):
    r"""The order violation of vectors in an ordered vector space
    (Vendrov et al. 2016, Eq. 2; Zhapa-Camacho and Hoehndorf 2023, Eq. 6):
    :math:`\lVert \max(0, v) \rVert_p`.

    :param direction: ``"reversed"`` (order embeddings): :math:`a \preceq b` iff
        :math:`a_i \ge b_i` for all :math:`i`, the origin is the top element,
        :math:`v = b - a`. ``"product"`` (CatE): :math:`a \preceq b` iff :math:`a_i \le b_i`,
        :math:`v = a - b`.
    :param p: Order of the norm. For :math:`p = 2` the square root is taken of the sum plus
        :math:`10^{-24}`, which keeps its gradient finite at 0 (and changes the value by at
        most :math:`10^{-12}`).

    Part (vocabulary value): ``order violation``. Used by: ``cate``, ``order_embeddings``.
    """

    accepts = (OrderCone,)
    part = "order violation"
    floored = True

    def __init__(self, direction="reversed", p=2, negative_form="hinge"):
        super().__init__(False, negative_form)
        if direction not in ("reversed", "product"):
            raise ValueError("direction must be 'reversed' or 'product'.")
        self.direction = direction
        self.p = p

    def _norm(self, x):
        x = th.relu(x)
        if self.p == 2:
            return (th.sqrt((x ** 2).sum(dim=-1) + 1e-24) - 1e-12).clamp(min=0.0)
        return th.linalg.vector_norm(x, ord=self.p, dim=-1)

    def _diff(self, a, b):
        return b.centre - a.centre if self.direction == "reversed" else a.centre - b.centre

    def raw(self, a, b):
        return self._norm(self._diff(a, b)).unsqueeze(-1)

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        return self._norm(self._diff(a, b) + margin).unsqueeze(-1)

    def violation(self, a, b):
        """The violation of plain vectors ``a`` below ``b`` (tensors, not regions), shape of
        the vectors without the last dimension."""
        return self._norm(b - a if self.direction == "reversed" else a - b)


# -- boxes: volumes -----------------------------------------------------------------------
def neg_log1mexp(energy, floor=1e-12):
    """:math:`-\\log(1 - e^{-E})` for :math:`E \\ge 0` (the cost of a missing edge whose
    presence has energy :math:`E`), computed stably; :math:`E` is clamped below at
    ``floor``."""
    e = th.clamp(energy, min=floor)
    small = e < math.log(2.0)
    out_small = th.log(-th.expm1(-th.where(small, e, th.ones_like(e))))
    out_large = th.log1p(-th.exp(-th.where(small, th.ones_like(e), e)))
    return -th.where(small, out_small, out_large)


class LogVolumeRatio(_LogProbability, Bounds):
    r"""The negative log of the conditional probability :math:`P(b \mid a)` of two boxes as
    the cost of :math:`a \sqsubseteq b`, and :math:`-\log(1 - P)` as the cost of a pair that
    is not ordered. This is *one* formula with two ways of getting the probability; the other
    volume-ratio measures are different formulas
    (:class:`~mowl.representation.library.boxrel_measures.VolumeShare`: ``1 - ratio``;
    :class:`~mowl.representation.library.shapes_measures.GumbelVolumeRatio`: ``(1 - ratio)^p``).

    **Hard boxes** (``intersection_temp=None``, default; the box lattice, Vilnis et al. 2018):
    :math:`P(b \mid a) = \mathrm{vol}(a \cap b) / \mathrm{vol}(a)` under the uniform
    measure. For disjoint boxes the probability has no gradient, so the lower bound
    :math:`\mathrm{vol}(a) + \mathrm{vol}(b) - \mathrm{vol}(a \vee b) \le
    \mathrm{vol}(a \wedge b)` (Section 4.3) is used for it, plus a penalty :math:`w g` with
    :math:`g` the part of the join that the two boxes do not cover (a term of mOWL, not of the
    paper: it keeps a gradient for a stated pair that is disjoint).

    **Smoothed boxes** (``intersection_temp`` a number; Gumbel boxes, Dasgupta et al. 2020,
    and binary code boxes, Zhang et al. 2022). In every dimension :math:`i`

    .. math:: F_i = \frac{m_\nu(\mu^\vee_i - \mu^\wedge_i)}
                         {m_\nu(a^+_i - a^-_i)}, \quad
              \mu^\wedge = \mathrm{LSE}_\tau(a^-, b^-),\
              \mu^\vee = -\mathrm{LSE}_\tau(-a^+, -b^+),

    with :math:`\mathrm{LSE}_\tau(x, y) = \tau \log(e^{x/\tau} + e^{y/\tau})` and the
    smoothed side length :math:`m_\nu(x) = \nu \log(1 + e^{x/\nu - s})`. The cost of
    :math:`a \sqsubseteq b` is the energy :math:`E = -\log P(b \mid a) = \sum_i -\log F_i`;
    the cost of a pair that is not ordered is :math:`-\log(1 - e^{-E})`. With
    :math:`\tau = \nu = 0` this is the hard box score without the disjointness term.
    If the regions carry codes (:class:`~.hierarchy_regions.CodedBox`), :math:`E` weights the
    dimensions: per-class codes :math:`\sum_i a^\diamond_i b^\diamond_i (-\log F_i)`; global
    code words :math:`\min_j \sum_i B_{ij} (-\log F_i)`.

    Used by: ``box_lattice`` (hard), ``gumbel_box`` and ``bcbox`` (smoothed).

    :param intersection_temp: ``None``: hard boxes. Otherwise :math:`\tau` (the Gumbel scale
        :math:`\beta` of Dasgupta et al.); ``0`` is the hard limit of the smoothed score.
    :param volume_temp: Smoothed boxes: :math:`\nu`; default :math:`\tau`.
    :param gumbel_shift: Smoothed boxes: :math:`s = 2\gamma` (Euler's constant: the softplus
        approximation of the expected side length of a Gumbel box, Dasgupta et al.
        Section 4.2) if ``True``, :math:`s = 0` (the volume temperature of Zhang et al.) if
        ``False``.
    :param tail: Smoothed boxes: how :math:`\log m_\nu` is computed where
        :math:`e^{x/\nu - s}` underflows: ``"exact"`` (:math:`z - \tfrac12 e^z`), ``"floor"``
        (:math:`\log(\cdot + 10^{-30})`, the earlier guard of the Gumbel box module) or
        ``"linear"`` (:math:`z` below :math:`-15`, the guard of the BCBox code and of the
        Gumbel box module).
    :param energy_floor: Smoothed boxes: smallest energy given to a negative pair (default
        :math:`10^{-12}`, the guard of the binary code boxes and of the Gumbel boxes;
        :math:`10^{-9}` is the earlier guard of the Gumbel box module).
    :param eps: Hard boxes: :math:`\varepsilon` inside the logarithms.
    :param disjoint_weight: Hard boxes: :math:`w`; default :math:`-\log \varepsilon`.
    :param zero_at_inclusion: Hard boxes: ``True`` (default): the cost of an included box is
        exactly 0 (the constant :math:`\log(1 + \varepsilon)` is added); ``False``: the
        guard of the earlier code, :math:`-\log(1 + \varepsilon)`.
    :param dtype: The precision of the computation (float64 in the published modules of the
        box lattice and the Gumbel box; ``None``: the regions' own, as in BCBox).

    The other relations (``meets``, ``empty``, ``nonempty``, the intersection forms) are those
    of :class:`~mowl.representation.Bounds`; ``apart`` is the conditional probability, which
    is (nearly) 0 when the boxes are (nearly) disjoint.
    """

    accepts = (Box,)
    exact_intersection = True

    @staticmethod
    def probability(energy):
        """:math:`P = e^{-E}`: the probability of an edge from its energy."""
        return th.exp(-energy)

    def __init__(self, intersection_temp=None, volume_temp=None, *, eps=1e-6,
                 disjoint_weight=None, zero_at_inclusion=True, gumbel_shift=True,
                 tail="exact", energy_floor=1e-12, dtype=th.float64, separation="any"):
        super().__init__(separation)
        self.smoothed = intersection_temp is not None
        if not self.smoothed and volume_temp is not None:
            raise ValueError("volume_temp is a parameter of smoothed boxes: pass "
                             "intersection_temp as well (0 for the hard limit).")
        if tail not in ("exact", "floor", "linear"):
            raise ValueError("tail must be 'exact', 'floor' or 'linear'.")
        if self.smoothed and (intersection_temp < 0 or (volume_temp is not None
                                                        and volume_temp < 0)):
            raise ValueError("Temperatures must be non-negative.")
        self.part = "smoothed volume ratio" if self.smoothed else "volume ratio"
        # hard boxes
        self.eps = float(eps)
        self.disjoint_weight = -math.log(self.eps) if disjoint_weight is None \
            else float(disjoint_weight)
        self.zero_at_inclusion = bool(zero_at_inclusion)
        # smoothed boxes
        self.tau = None if not self.smoothed else float(intersection_temp)
        self.nu = None if not self.smoothed else self.tau if volume_temp is None \
            else float(volume_temp)
        self.shift = 2.0 * EULER_GAMMA if gumbel_shift else 0.0
        self.tail = tail
        self.energy_floor = float(energy_floor)
        self.dtype = dtype

    # -- hard boxes -------------------------------------------------------------------------
    def conditional(self, a, b):
        """Hard boxes: ``(p, gap)``, ``(B, 1)`` each: the estimate of :math:`P(b \\mid a)`
        (the exact value for intersecting boxes, the lower bound, which is :math:`\\le 0`, for
        disjoint ones) and the uncovered fraction :math:`g` of the join (0 if they
        intersect)."""
        if self.smoothed:
            raise ConstructorUnavailable("conditional() is the probability of hard boxes; "
                                         "smoothed boxes have an energy().")
        self.check(a, b)
        lo_a, hi_a = a.lower.to(self.dtype), a.upper.to(self.dtype)
        lo_b, hi_b = b.lower.to(self.dtype), b.upper.to(self.dtype)
        tiny = th.finfo(self.dtype).tiny
        vol_a, vol_b = volume(lo_a, hi_a), volume(lo_b, hi_b)
        lo_m, hi_m = meet(lo_a, hi_a, lo_b, hi_b)
        lo_j, hi_j = join(lo_a, hi_a, lo_b, hi_b)
        intersect = (hi_m > lo_m).all(dim=-1)
        vol_j = volume(lo_j, hi_j).clamp(min=tiny)
        bound = vol_a + vol_b - vol_j
        joint = th.where(intersect, volume(lo_m, hi_m), bound)
        gap = th.where(intersect, th.zeros_like(vol_j), -bound / vol_j)
        return (joint / vol_a.clamp(min=tiny)).unsqueeze(-1), gap.unsqueeze(-1)

    # -- smoothed boxes ---------------------------------------------------------------------
    def _lse(self, x, y):
        if self.tau == 0:
            return th.maximum(x, y)
        return self.tau * th.logaddexp(x / self.tau, y / self.tau)

    def log_side(self, x):
        """Smoothed boxes: :math:`\\log m_\\nu(x)`, elementwise."""
        if self.nu == 0:
            return th.log(th.clamp(x, min=0.0))
        z = x / self.nu - self.shift
        if self.tail == "floor":
            inner = th.log(F.softplus(z) + 1e-30)
        else:
            safe = th.clamp(z, min=-15.0)
            tail = z + th.log1p(-0.5 * th.exp(z)) if self.tail == "exact" else z
            inner = th.where(z > -15.0, th.log(F.softplus(safe)), tail)
        return math.log(self.nu) + inner

    def neg_log_scores(self, a, b):
        """Smoothed boxes: :math:`-\\log F_i` of the dimensions, ``(B, d)``."""
        self.check(a, b)
        lo_a, hi_a = self._cast(a.lower), self._cast(a.upper)
        lo_b, hi_b = self._cast(b.lower), self._cast(b.upper)
        meet_min = self._lse(lo_a, lo_b)
        meet_max = -self._lse(-hi_a, -hi_b)
        return -(self.log_side(meet_max - meet_min) - self.log_side(hi_a - lo_a))

    def _cast(self, x):
        return x if self.dtype is None else x.to(self.dtype)

    @staticmethod
    def _weighted(codes, nls):
        # F^0 = 1: a dimension whose code is exactly 0 is ignored even where -log F is inf.
        w = codes * nls
        return th.where(codes == 0, th.zeros_like(w), w)

    def energy(self, a, b):
        """Smoothed boxes: :math:`E = -\\log P(b \\mid a)`, ``(B, 1)``, with the codes of
        the regions."""
        if not self.smoothed:
            raise ConstructorUnavailable("energy() is the score of smoothed boxes; hard "
                                         "boxes have a conditional().")
        nls = self.neg_log_scores(a, b)
        code_a, code_b = getattr(a, "code", None), getattr(b, "code", None)
        words = getattr(a, "global_code", None)
        if code_a is not None and code_b is not None:
            e = self._weighted(self._cast(code_a) * self._cast(code_b), nls).sum(dim=-1)
        elif words is not None:
            e = self._weighted(self._cast(words), nls.unsqueeze(-1)).sum(dim=-2) \
                .min(dim=-1).values
        else:
            e = nls.sum(dim=-1)
        return e.unsqueeze(-1)

    # -- the costs ---------------------------------------------------------------------------
    def inside(self, a, b, margin=0.0):
        self._no_margin(margin, "The log volume ratio")
        if self.smoothed:
            return self.energy(a, b).clamp(min=0.0).to(a.centre.dtype)
        p, gap = self.conditional(a, b)
        if self.zero_at_inclusion:
            # exactly 0 for an included pair (p = 1), whatever the rounding of p
            log_term = -th.log((p.clamp(min=0.0, max=1.0) + self.eps) / (1.0 + self.eps))
        else:
            log_term = -th.log(p.clamp(min=0.0) + self.eps)
        return (log_term + self.disjoint_weight * gap).to(a.centre.dtype)

    def negative(self, a, b, margin=0.0):
        """:math:`-\\log(1 - P(b \\mid a))`, the cross entropy with target 0 (hard boxes: the
        estimate is clamped to :math:`[0, 1 - \\varepsilon]`)."""
        if self.smoothed:
            return neg_log1mexp(self.energy(a, b), self.energy_floor).to(a.centre.dtype)
        p, _ = self.conditional(a, b)
        return (-th.log1p(-p.clamp(min=0.0, max=1.0 - self.eps))).to(a.centre.dtype)

    def apart(self, a, b, margin=0.0):
        """:math:`P(b \\mid a)`: small when the boxes are (nearly) disjoint."""
        self._no_margin(margin, "The log volume ratio")
        if self.smoothed:
            return th.exp(-self.energy(a, b)).to(a.centre.dtype)
        p, _ = self.conditional(a, b)
        return p.clamp(min=0.0).to(a.centre.dtype)


# -- cones, disks, points -----------------------------------------------------------------
class ConeAngle(EnergyMeasure):
    r"""The angle of a point against the aperture of a geodesic cone (Ganea et al. 2018,
    Eq. 33): :math:`\max(0, \Xi(b, a) - \psi(b))` with :math:`\Xi` the angle between the
    axis of the cone of the parent :math:`b` and the geodesic to the child :math:`a`; zero
    iff :math:`a` lies in the cone of :math:`b`. Poincare ball only.

    Part (vocabulary value): ``angle``. Used by: ``hyperbolic_cones``.
    """

    accepts = (ConePoint,)
    part = "angle"
    floored = True

    def __init__(self, signed=False, negative_form="hinge", apex="inside"):
        super().__init__(signed, negative_form)
        if apex not in ("inside", "formula"):
            raise ValueError("apex must be 'inside' or 'formula'.")
        #: ``"inside"`` (default): the apex of a cone lies in it, so :math:`C \\sqsubseteq C`
        #: costs nothing. ``"formula"``: the published formula, whose angle is undefined at
        #: the apex and is read as the clamped arccosine there (a positive cost; the method
        #: ``hyperbolic_cones`` and its module).
        self.apex = apex

    def raw(self, a, b):
        # "formula": the arccosine of the clamped cosine, as the published formula is evaluated
        angle = OriginLight(1.0).axis_angle(b.manifold, b.centre, a.centre,
                                            cos_clamp=1e-6 if self.apex == "formula" else 0.0)
        if self.apex == "inside":
            apex = th.linalg.norm(a.centre - b.centre, dim=-1) < 1e-7
            angle = th.where(apex, th.zeros_like(angle), angle)
        return (angle - b.aperture).unsqueeze(-1)


class ShadowCone(EnergyMeasure):
    """The distance of a point to the shadow cone of a parent (Yu et al. 2024, Section 4):
    negative inside the cone, positive outside.

    :param formulation: ``"penumbral-ball"``, ``"penumbral-half-space"``, ``"umbral-ball"``
        or ``"umbral-half-space"``; the half-space ones need points of an
        :class:`~mowl.representation.library.manifolds.UpperHalfSpace`, the others of a Poincare ball.
    :param radius, horosphere_height, gamma3: See
        :class:`~mowl.representation.library.cone_regions.Shadow`.

    Part (vocabulary value): ``angle``. Used by: ``shadow_cones``.
    """

    accepts = (ManifoldPoint,)
    part = "angle"

    def __init__(self, formulation="umbral-half-space", radius=0.05, horosphere_height=20.0,
                 gamma3=1e-4, signed=False, negative_form="hinge"):
        if formulation not in FORMULATIONS:
            raise ValueError(f"formulation must be one of {FORMULATIONS}.")
        super().__init__(signed, negative_form)
        self.formulation = formulation
        self.radius = radius
        self.horosphere_height = horosphere_height
        self.gamma3 = gamma3

    def raw(self, a, b):
        half_space = isinstance(a.manifold, UpperHalfSpace)
        if half_space != self.formulation.endswith("half-space"):
            raise ConstructorUnavailable(f"The formulation {self.formulation!r} does not "
                                         f"live on {type(a.manifold).__name__}.")
        light, umbral = light_for(self.formulation, self.radius, self.horosphere_height)
        cone = Shadow(b.centre, b.manifold, b.ids, light, umbral, self.radius, self.gamma3)
        return cone.signed_distance(a.centre).unsqueeze(-1)


class Protrusion(EnergyMeasure):
    r"""The protrusion of a disk out of another (Suzuki et al. 2019, Eq. 9):
    :math:`l = d(x_b, x_a) - r_b + r_a`, :math:`\le 0` iff the disk of the parent :math:`b`
    contains that of the child :math:`a`. The distance is that of the manifold the disks live
    on, so the same measure serves every space (and, on the Euclidean one, balls: with
    non-negative radii it is the measure of
    :class:`~mowl.representation.CentreDistance`). Balls of the Euclidean class models
    (:class:`~mowl.representation.Ball`) are disks on the Euclidean space.

    Part (vocabulary value): ``protrusion``. Used by: ``disk_embeddings``.
    """

    accepts = (Disk, Ball)
    part = "protrusion"
    _euclidean = Euclidean()

    def raw(self, a, b):
        manifold = getattr(a, "manifold", None) or self._euclidean
        distance = manifold.distance(b.centre, a.centre).reshape(-1, 1)
        return distance - b.radius.reshape(-1, 1) + a.radius.reshape(-1, 1)


class DistanceDepth(HierarchyMeasure):
    r"""Distance and depth of points (HiT, He et al. 2024, Eq. 6): the cost of :math:`a
    \sqsubseteq b` is :math:`d(a, b) + \lambda (\lVert b \rVert - \lVert a \rVert)`, the
    negated subsumption score, with :math:`\lVert x \rVert = d(x, 0)` the depth of a point.

    This is a ranking score, not a satisfiability cost: it is not zero where the relation
    holds and it can be negative (published reading; there is no other). Pairs that are not
    ordered are scored by the same function. The *training* loss of HiT is a triplet loss,
    available as :meth:`triplet`.

    :param centri_weight: :math:`\\lambda` (``centri_weight`` of HiT; 0 uses the distance
        only).

    Part (vocabulary value): ``distance and depth``. Used by: ``hit``, ``ont``.
    """

    accepts = (ManifoldPoint,)
    part = "distance and depth"
    reading = "score"

    def __init__(self, centri_weight=1.0):
        self.centri_weight = float(centri_weight)

    def inside(self, a, b, margin=0.0, centri_weight=None):
        """The cost :math:`d(a, b) + \\lambda (\\lVert b \\rVert - \\lVert a \\rVert)`, ``(B, 1)``;
        ``centri_weight`` replaces :math:`\\lambda` for this call (HiT selects it on validation
        data after training)."""
        self.check(a, b)
        self._no_margin(margin, "The distance-and-depth score")
        lam = self.centri_weight if centri_weight is None else centri_weight
        gap = depth(b.manifold, b.centre) - depth(a.manifold, a.centre)
        return (a.manifold.distance(a.centre, b.centre) + lam * gap).unsqueeze(-1)

    def ranking_scores(self, query, candidates, candidates_are_children=False,
                       centri_weight=None, dtype=th.float64):
        """The scores (higher: more plausible) of every candidate against every query, the
        negated costs of :meth:`inside`, ``(n, m)``: ``query`` ``(n, d)`` and ``candidates``
        ``(m, d)`` are points of the ball of the class model (``ManifoldPoint`` centres); the
        candidates are the subclasses of the queries when ``candidates_are_children``, their
        superclasses otherwise. This is the evaluation of HiT/OnT on thousands of candidates:
        the distance matrix is computed in ``dtype`` without an ``(n, m, d)`` tensor."""
        lam = self.centri_weight if centri_weight is None else centri_weight
        manifold = query.manifold
        q = query.centre
        q_norm = depth(manifold, q.to(dtype)).unsqueeze(1)
        dist = manifold.pairwise_distance(q, candidates.centre, dtype=dtype)
        c_norm = depth(manifold, candidates.centre.to(dtype)).unsqueeze(0)
        if candidates_are_children:
            return -(dist + lam * (q_norm - c_norm))
        return -(dist + lam * (c_norm - q_norm))

    def negative(self, a, b, margin=0.0):
        return self.inside(a, b)

    def clustering(self, a, b, n, margin):
        """HiT Eq. 3: :math:`\\max(d(a, b) - d(a, n) + \\alpha, 0)` for a child ``a``, its
        parent ``b`` and a negative parent ``n``; ``(B, 1)``."""
        m = a.manifold
        gap = m.distance(a.centre, b.centre) - m.distance(a.centre, n.centre)
        return th.relu(gap + margin).unsqueeze(-1)

    def centripetal(self, a, b, margin):
        """HiT Eq. 4: :math:`\\max(\\lVert b \\rVert - \\lVert a \\rVert + \\beta, 0)`;
        ``(B, 1)``."""
        gap = depth(b.manifold, b.centre) - depth(a.manifold, a.centre)
        return th.relu(gap + margin).unsqueeze(-1)

    def triplet(self, a, b, n, clustering_margin, centripetal_margin, clustering_weight=1.0,
                centripetal_weight=1.0):
        """:math:`L_\\prec` (HiT Eq. 5, OnT Eq. 5): clustering plus centripetal loss."""
        cluster = self.clustering(a, b, n, clustering_margin)
        centri = self.centripetal(a, b, centripetal_margin)
        return clustering_weight * cluster + centripetal_weight * centri


# -- registration ------------------------------------------------------------------------
register_measure("order violation", default_for=(OrderCone,))(OrderViolation)
register_measure("log volume ratio", default_for=(CornerBox,))(LogVolumeRatio)
DEFAULT_MEASURE[CodedBox] = lambda: LogVolumeRatio(0.01)
register_measure("angle", default_for=(ConePoint,))(ConeAngle)
register_measure("shadow cone")(ShadowCone)
register_measure("protrusion", default_for=(Disk,))(Protrusion)
register_measure("distance and depth", default_for=(ManifoldPoint,))(DistanceDepth)
