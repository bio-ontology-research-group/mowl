r"""Measures of the shapes group.

=============================  =======================  ======================================
Measure                        Regions                  Vocabulary
=============================  =======================  ======================================
:class:`OctagonBounds`         octagon products         bounds per coordinate
:class:`PrismBounds`           prisms                   bounds per coordinate
:class:`EllipsoidMeasure`      ellipsoids and points    centre distance
:class:`CosineOrder`           vectors                  similarity
:class:`LinkedMeasure`         linked regions           (that of the first space)
:class:`GumbelVolumeRatio`      (soft) boxes             smoothed volume ratio
:class:`PointToRegion`         point and (soft) box     point-to-region distance
:class:`PointDistance`         points                   centre distance
:class:`ByRegion`              any                      (that of the first rule)
=============================  =======================  ======================================

Every measure follows :mod:`mowl.representation.measures`: costs have shape ``(B, 1)``,
lower is more true, a positive margin is strict. Some add a method for what a method of
the group needs and the core does not have: ``member`` (a point in a region), ``outside``
(a region sticks out of another by a margin).
"""

import torch as th
import torch.nn.functional as F

from mowl.nn.el.concept2box import losses as C2B
from mowl.nn.el.octagon import geometry as G
from mowl.nn.el.octagon import losses as OL
from mowl.representation.measures import RegionMeasure
from mowl.representation.methods import register_measure
from mowl.representation.regions import Box, ConstructorUnavailable
from mowl.representation.library.shapes_regions import (
    Point, OctagonProduct, Prism, Ellipsoid, Vector, Linked, SoftBox)

__all__ = ["OctagonBounds", "PrismBounds", "EllipsoidMeasure", "CosineOrder", "LinkedMeasure",
           "GumbelVolumeRatio", "PointToRegion", "PointDistance", "ByRegion"]


def _col(x):
    return x.reshape(-1, 1)


def _norm(x):
    return th.linalg.norm(x, dim=-1, keepdim=True)


def member_cost(measure, a, b, margin=0.0):
    """``measure.member`` if the measure has it, else ``measure.inside``."""
    method = getattr(measure, "member", None)
    return method(a, b, margin) if method is not None else measure.inside(a, b, margin)


# -- octagons ------------------------------------------------------------------------------

class OctagonBounds(RegionMeasure):
    r"""Products of octagons compared by their bounds. :math:`A \subseteq B` iff
    :math:`T(h_A) + m \le h_B` on every direction of every factor (:math:`T` tightens the
    bounds, so that the bounds of :math:`A` are its support function); the cost is the
    norm of the violations, taking for each of the four facet axes the larger of the two
    opposite directions. On boxes it lies between the cost of
    :class:`~mowl.representation.measures.Bounds` and :math:`\sqrt 5` times it.

    * ``apart``: the smallest width of the intersection is at most :math:`-m`.
    * ``meets``: the intersection has width at least :math:`m` on every axis.
    * ``empty``: the smallest width, zero iff the interior is empty.
    """

    accepts = (OctagonProduct,)
    part = "bounds per coordinate"

    def inside(self, a, b, margin=0.0):
        return _col(OL.inclusion_loss(a.bounds, b.bounds, margin))

    def meets(self, a, b, margin=0.0):
        widths = G.widths(th.minimum(a.bounds, b.bounds))
        return _norm(th.relu(margin - widths).reshape(widths.shape[0], -1))

    def apart(self, a, b, margin=0.0):
        return _col(OL.disjoint_loss(a.bounds, b.bounds, margin))

    def outside(self, a, b, margin=0.0):
        """Zero iff some bound of ``a`` exceeds that of ``b`` by at least ``margin``."""
        return _col(OL.non_inclusion_loss(a.bounds, b.bounds, margin))

    def empty(self, a, margin=0.0):
        return _col(OL.interior_empty_loss(a.bounds))

    def nonempty(self, a, epsilon=0.0):
        return _col(th.relu(epsilon - G.product_emptiness_slack(a.bounds)))


# -- prisms ----------------------------------------------------------------------------------

class PrismBounds(RegionMeasure):
    r"""Prisms compared by their bounds: the three support values of the triangle and the
    box coordinates, as the norm of the violations of :math:`h_A \le h_B` and of the box
    inclusion.

    * ``apart``: the triangles are disjoint or the boxes are separated in some coordinate.
    * ``meets``: the triangle part of the intersection has positive support sum and the
      boxes overlap in every coordinate.
    * ``empty``: ``"whole"`` (default) the norm of the triangle's support sum and the box
      half-widths; ``"triangle"`` only the support sum, as the prism module has it
      (the cost of a class that cannot reach zero with a positive margin, and it ignores
      the box).
    * ``meets_shrunk``: the negative reading of the published loss, see its docstring.

    The half-widths of the box part are read as they are: a *negative* half-width is a box that
    is empty in that coordinate (the signed intersection of
    :class:`~mowl.representation.library.structural_regions.DefinedClasses`), and the costs
    treat it as such (an empty region is not included in a smaller one, does not meet, is
    apart from everything). The regions of the class models have non-negative half-widths,
    for which nothing changes.

    :param empty: ``"whole"`` or ``"triangle"``.
    """

    accepts = (Prism,)
    part = "bounds per coordinate"

    def __init__(self, empty="whole"):
        if empty not in ("whole", "triangle"):
            raise ValueError("empty must be 'whole' or 'triangle'.")
        self.empty_form = empty

    def inside(self, a, b, margin=0.0):
        tri = th.relu(a.supports - b.supports + margin)
        gap = th.abs(a.centre - b.centre) + a.offset - b.offset
        return _norm(th.cat([tri, th.relu(gap + margin)], dim=-1))

    def meets(self, a, b, margin=0.0):
        tri = th.relu(-th.minimum(a.supports, b.supports).sum(dim=-1, keepdim=True) + margin)
        return _norm(th.cat([tri, self._box_meets(a, b, margin)], dim=-1))

    @staticmethod
    def _box_meets(a, b, margin):
        gap = th.abs(a.centre - b.centre) - a.offset - b.offset
        return th.relu(gap + margin)

    def meets_shrunk(self, a, b, margin=0.0):
        r"""The negative loss of the prism module for :math:`C \sqsubseteq \exists r.D`:
        the support values of ``a`` shrunk by twice its inradius, against those of ``b``,
        and the boxes with ``-margin`` slack. Zero when ``a`` is a point of the triangle
        part inside ``b``'s supports. It is *not* an overlap test (a triangle of small
        inradius overlapping ``b`` can have a positive cost), and it is kept as the
        published reading."""
        radius = th.relu(a.supports.sum(dim=-1, keepdim=True) / 3.0)
        tri = th.relu(a.supports - 2.0 * th.abs(radius) - b.supports + margin)
        return _norm(th.cat([tri, self._box_meets(a, b, margin)], dim=-1))

    def apart(self, a, b, margin=0.0):
        tri = th.relu(th.minimum(a.supports, b.supports).sum(dim=-1) + margin)
        overlap = a.offset + b.offset - th.abs(a.centre - b.centre)
        box = th.relu(overlap + margin).amin(dim=-1)
        return _col(th.minimum(tri, box))

    def empty(self, a, margin=0.0):
        tri = th.relu(a.supports.sum(dim=-1, keepdim=True) + margin)
        if self.empty_form == "triangle":
            return tri
        return _norm(th.cat([tri, th.relu(a.offset)], dim=-1))

    def nonempty(self, a, epsilon=0.0):
        size = th.minimum(a.supports.sum(dim=-1), 2.0 * a.offset.amin(dim=-1))
        return _col(th.relu(epsilon - size))


# -- ellipsoids and vectors --------------------------------------------------------------------

class EllipsoidMeasure(RegionMeasure):
    r"""Ellipsoids and points. Zero costs mean:

    * point in ellipsoid (``member``, ``inside`` of a point): :math:`\sum_j ((x_j -
      c_j)/b_j)^2 \le r^2`; the cost is the excess, exactly as in EIKE.
    * point in point (``inside``): the two are the same point; the cost is the squared
      distance.
    * ellipsoid in ellipsoid: with ``containment="bounding"`` (default) the sufficient
      condition :math:`\lVert (c_A - c_B)/b_B \rVert + r_A \max_j (b_{A,j}/b_{B,j}) \le r_B`,
      which implies containment; with ``"published"`` the formula as printed in the
      EIKE paper, :math:`\sum_j (c_{A,j}/b_{A,j} - c_{B,j}/b_{B,j})^2 + r_A^2 - r_B^2 \le
      0`, which does not (it compares the centres divided by two different axes, and
      squares the distance).
    * ``apart``: the enclosing balls of radius :math:`r \max_j b_j` do not intersect
      (a sufficient condition), the point lies outside, or the points differ by ``margin``.

    There is no conjunction (the intersection of ellipsoids is not an ellipsoid).
    """

    accepts = (Ellipsoid, Point)
    part = "centre distance"
    exact_intersection = False

    def __init__(self, containment="bounding"):
        if containment not in ("bounding", "published"):
            raise ValueError("containment must be 'bounding' or 'published'.")
        self.containment = containment

    @staticmethod
    def _excess(x, e):
        return (((x - e.centre) / e.axes) ** 2).sum(dim=-1, keepdim=True) - e.radius ** 2

    def member(self, a, b, margin=0.0):
        """Zero iff the individual ``a`` lies in the ellipsoid ``b`` (``margin`` room): a
        point, or an ellipsoid of no extent."""
        if not isinstance(a, Point):
            return self.inside(a, b, margin)
        if not isinstance(b, Ellipsoid):
            raise TypeError("member compares a Point with an Ellipsoid.")
        return th.relu(self._excess(a.centre, b) + margin)

    def inside(self, a, b, margin=0.0):
        if isinstance(a, Point):
            if isinstance(b, Point):
                squared = ((a.centre - b.centre) ** 2).sum(dim=-1, keepdim=True)
                return th.relu(squared + margin)
            return self.member(a, b, margin)
        if isinstance(b, Point):
            raise ConstructorUnavailable("An ellipsoid is not inside a point.")
        if self.containment == "published":
            d = ((a.centre / a.axes - b.centre / b.axes) ** 2).sum(dim=-1, keepdim=True)
            return th.relu(d + a.radius ** 2 - b.radius ** 2 + margin)
        d = _norm((a.centre - b.centre) / b.axes)
        stretch = (a.axes / b.axes).amax(dim=-1, keepdim=True)
        return th.relu(d + a.radius * stretch - b.radius + margin)

    def meets(self, a, b, margin=0.0):
        raise ConstructorUnavailable("EllipsoidMeasure has no overlap test.")

    def apart(self, a, b, margin=0.0):
        if isinstance(a, Point):
            if isinstance(b, Point):
                return th.relu(margin - _norm(a.centre - b.centre))
            return th.relu(-self._excess(a.centre, b) + margin)
        if isinstance(b, Point):
            return th.relu(-self._excess(b.centre, a) + margin)
        reach_a = a.radius * a.axes.amax(dim=-1, keepdim=True)
        reach_b = b.radius * b.axes.amax(dim=-1, keepdim=True)
        return th.relu(reach_a + reach_b - _norm(a.centre - b.centre) + margin)

    def empty(self, a, margin=0.0):
        return th.zeros_like(a.centre[:, :1]) if isinstance(a, Point) else a.radius

    def nonempty(self, a, epsilon=0.0):
        return th.relu(epsilon - self.empty(a))


class CosineOrder(RegionMeasure):
    r"""Vectors compared by direction and length: :math:`A \subseteq B` costs
    :math:`1 - \cos(a, b)` plus a term for the length, which asks the narrower name to have
    the shorter vector (EIKE's intensional score). ``member`` is the cosine distance alone
    (an instance has no length of its own to compare).

    :param norm_term: ``"hinge"`` (default): :math:`[\lVert a\rVert - \lVert b\rVert]_+`,
        the cost stays non-negative; ``"signed"``: :math:`\lVert a\rVert - \lVert b\rVert`
        as in the EIKE code (can be negative, so the whole cost can); ``"none"``.
    """

    accepts = (Vector,)
    part = "similarity"
    exact_intersection = False

    def __init__(self, norm_term="hinge"):
        if norm_term not in ("hinge", "signed", "none"):
            raise ValueError("norm_term must be 'hinge', 'signed' or 'none'.")
        self.norm_term = norm_term

    @staticmethod
    def _cos(a, b):
        return _col(F.cosine_similarity(a.centre, b.centre, dim=-1))

    def member(self, a, b, margin=0.0):
        return th.relu(1 - self._cos(a, b) + margin)

    def inside(self, a, b, margin=0.0):
        cost = 1 - self._cos(a, b)
        diff = _col(a.centre.norm(dim=-1) - b.centre.norm(dim=-1))
        if self.norm_term == "signed":
            return cost + diff + margin
        if self.norm_term == "hinge":
            cost = cost + th.relu(diff)
        return th.relu(cost + margin)

    def meets(self, a, b, margin=0.0):
        return th.relu(margin - self._cos(a, b))

    def apart(self, a, b, margin=0.0):
        return th.relu(self._cos(a, b) + margin)

    def empty(self, a, margin=0.0):
        return _col(a.centre.norm(dim=-1))

    def nonempty(self, a, epsilon=0.0):
        return th.relu(epsilon - self.empty(a))


class LinkedMeasure(RegionMeasure):
    r"""Regions of two linked spaces: the cost is the cost in the first space plus ``weight``
    times the cost in the second, where both regions have one (a region built by a relation
    has none, and costs nothing there). ``member`` uses the ``member`` of both spaces.

    :param first, second: The measures of the two spaces.
    :param weight: The weight :math:`\alpha` of the second space.
    """

    accepts = (Linked,)
    exact_intersection = False

    def __init__(self, first, second, weight=1.0):
        self.first, self.second = first, second
        self.weight = float(weight)

    @property
    def part(self):
        return self.first.part

    def _both(self, f, g, a, b, margin):
        cost = f(a.first, b.first, margin)
        if a.second is not None and b.second is not None:
            cost = cost + self.weight * g(a.second, b.second, 0.0)
        return cost

    def inside(self, a, b, margin=0.0):
        return self._both(self.first.inside, self.second.inside, a, b, margin)

    def member(self, a, b, margin=0.0):
        return self._both(lambda x, y, m: member_cost(self.first, x, y, m),
                          lambda x, y, m: member_cost(self.second, x, y, m), a, b, margin)

    def meets(self, a, b, margin=0.0):
        return self._both(self.first.meets, self.second.meets, a, b, margin)

    def apart(self, a, b, margin=0.0):
        return self._both(self.first.apart, self.second.apart, a, b, margin)

    def empty(self, a, margin=0.0):
        return self.first.empty(a.first)

    def nonempty(self, a, epsilon=0.0):
        return self.first.nonempty(a.first, epsilon)


# -- soft boxes and points ----------------------------------------------------------------------

class GumbelVolumeRatio(RegionMeasure):
    r"""Boxes compared by the ratio of their expected Gumbel volumes (Dasgupta et al. 2020;
    Concept2Box): :math:`\phi(A, B) = \mathrm{E}[\mathrm{Vol}(A \cap B)] /
    \mathrm{E}[\mathrm{Vol}(A)]`, near 1 when :math:`A` is inside :math:`B` and near 0
    when they are apart. The volumes are positive for every pair of boxes, so disjoint boxes
    still receive gradients; with :math:`\beta \to 0` they become hard volumes. This is one of
    three different volume-ratio formulas (the others:
    :class:`~mowl.representation.library.boxrel_measures.VolumeShare`, ``1 - ratio`` with
    softplus volumes, and
    :class:`~mowl.representation.library.hierarchy_measures.LogVolumeRatio`, ``-log ratio``);
    here the cost is a power of the deviation of :math:`\phi` from its target. Used by:
    ``concept2box``.

    * ``inside``: :math:`[1 - \phi + m]_+^p`; ``apart``: :math:`[\phi + m]_+^p`;
      ``meets``: :math:`[m - \phi]_+^p`.
    * ``empty``: the expected volume.

    :param beta: The Gumbel scale when the region has none (plain boxes).
    :param power: :math:`p`: ``1`` (default) or ``2`` (the squared deviations of the
        Concept2Box loss).
    """

    accepts = (Box,)
    part = "smoothed volume ratio"

    def __init__(self, beta=1.0, power=1):
        self.beta = float(beta)
        self.power = power

    def _beta(self, a):
        return getattr(a, "beta", self.beta)

    def ratio(self, a, b):
        """:math:`\\phi(A, B)`, ``(B, 1)``."""
        beta = self._beta(a)
        lo, hi = C2B.gumbel_intersection(a.lower, a.upper, b.lower, b.upper, beta)
        log = C2B.gumbel_log_volume(lo, hi, beta) - C2B.gumbel_log_volume(
            a.lower, a.upper, beta)
        return _col(th.exp(log))

    def _p(self, x):
        return th.relu(x) ** self.power

    def inside(self, a, b, margin=0.0):
        return self._p(1 - self.ratio(a, b) + margin)

    def meets(self, a, b, margin=0.0):
        return self._p(margin - self.ratio(a, b))

    def apart(self, a, b, margin=0.0):
        return self._p(self.ratio(a, b) + margin)

    def empty(self, a, margin=0.0):
        return _col(th.exp(C2B.gumbel_log_volume(a.lower, a.upper, self._beta(a))))

    def nonempty(self, a, epsilon=0.0):
        return th.relu(epsilon - self.empty(a))


class PointToRegion(RegionMeasure):
    r"""The distance of a point to a (soft) box (Concept2Box, Eq. 8 and 9): the :math:`L_1`
    distance to the box plus, for a point inside, :math:`\alpha` times its distance to the
    centre.

    * ``member`` / ``inside``: that distance, or with ``squash`` :math:`\sigma(d) - 1/2`.
    * ``apart``: :math:`[m - s]_+`, zero when the point is farther than ``margin``
      (the published negative loss of a class assertion).

    :param alpha: Weight of the distance to the centre of a point inside. The default ``0``
        makes the cost zero iff the point lies in the box; Concept2Box uses
        :math:`\alpha(c) = \alpha \cdot \mathrm{sigmoid}(1/\mathrm{Vol}(c))` with
        :math:`\alpha = 0.5`, which pulls points to the centres (``weighted=True``).
    :param beta: Gumbel scale of the volume in that weight, when the box has none.
    :param weighted: Whether the weight depends on the volume of the box (Eq. 9).
    :param squash: Whether to squash the distance as the paper's cross-view loss does.
    """

    accepts = (Point, Box)
    part = "point-to-region distance"

    def __init__(self, alpha=0.0, beta=1.0, weighted=True, squash=False):
        self.alpha, self.beta = float(alpha), float(beta)
        self.weighted = weighted
        self.squash = squash

    def distance(self, point, box):
        beta = getattr(box, "beta", self.beta)
        e = point.centre
        out = (th.relu(e - box.upper) + th.relu(box.lower - e)).sum(dim=-1)
        if self.alpha == 0.0:
            return _col(out)
        inside_point = th.minimum(box.upper, th.maximum(box.lower, e))
        dist_in = (box.centre - inside_point).abs().sum(dim=-1)
        if self.weighted:
            weight = C2B.inside_weight(C2B.gumbel_log_volume(box.lower, box.upper, beta),
                                       self.alpha)
        else:
            weight = self.alpha
        return _col(out + weight * dist_in)

    def _score(self, a, b):
        if not isinstance(a, Point) or not isinstance(b, Box):
            raise TypeError("PointToRegion compares a Point with a box.")
        d = self.distance(a, b)
        return th.sigmoid(d) - 0.5 if self.squash else d

    def member(self, a, b, margin=0.0):
        return th.relu(self._score(a, b) + margin)

    def inside(self, a, b, margin=0.0):
        return self.member(a, b, margin)

    def meets(self, a, b, margin=0.0):
        return self.member(a, b, margin)

    def apart(self, a, b, margin=0.0):
        return th.relu(margin - self._score(a, b))

    def empty(self, a, margin=0.0):
        return th.zeros_like(a.centre[:, :1])

    def nonempty(self, a, epsilon=0.0):
        return th.relu(epsilon - self.empty(a))


class PointDistance(RegionMeasure):
    r"""Points: ``inside`` (equal) costs the distance to the power ``power``; ``apart``
    costs :math:`[m - \lVert a - b \rVert]_+`.

    :param power: ``1`` (default) or ``2``.
    """

    accepts = (Point,)
    part = "centre distance"
    exact_intersection = False

    def __init__(self, power=1):
        self.power = power

    def _d(self, a, b):
        return _norm(a.centre - b.centre) ** self.power

    def inside(self, a, b, margin=0.0):
        return th.relu(self._d(a, b) + margin)

    meets = inside

    def apart(self, a, b, margin=0.0):
        return th.relu(margin - _norm(a.centre - b.centre))

    def empty(self, a, margin=0.0):
        return th.zeros_like(a.centre[:, :1])

    def nonempty(self, a, epsilon=0.0):
        return th.relu(epsilon - self.empty(a))


class ByRegion(RegionMeasure):
    """A measure that is another measure for another pair of region kinds (Concept2Box:
    a volume ratio for two boxes, a point-to-box distance for a point and a box, a distance
    for two points). The first rule whose two region classes match is used.

    :param rules: ``[(class of the first region, class of the second, measure), ...]``.
    """

    def __init__(self, rules):
        self.rules = list(rules)
        self.accepts = tuple(dict.fromkeys(c for r in self.rules for c in r[:2]))
        self.exact_intersection = self.rules[0][2].exact_intersection

    @property
    def part(self):
        return self.rules[0][2].part

    def _measure(self, a, b):
        for ca, cb, measure in self.rules:
            if isinstance(a, ca) and isinstance(b, cb):
                return measure
        raise TypeError(f"No rule for {type(a).__name__} against {type(b).__name__}.")

    def inside(self, a, b, margin=0.0):
        return self._measure(a, b).inside(a, b, margin)

    def member(self, a, b, margin=0.0):
        return member_cost(self._measure(a, b), a, b, margin)

    def meets(self, a, b, margin=0.0):
        return self._measure(a, b).meets(a, b, margin)

    def apart(self, a, b, margin=0.0):
        return self._measure(a, b).apart(a, b, margin)

    def empty(self, a, margin=0.0):
        return self._measure(a, a).empty(a, margin)

    def nonempty(self, a, epsilon=0.0):
        return self._measure(a, a).nonempty(a, epsilon)

    def intersection_inside(self, a, b, c, margin=0.0):
        return self._measure(a, b).intersection_inside(a, b, c, margin)

    def intersection_apart(self, a, b, c, margin=0.0):
        return self._measure(a, b).intersection_apart(a, b, c, margin)

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        return self._measure(a, b).intersection_nonempty(a, b, margin, epsilon)


register_measure("gumbel volume ratio", default_for=(SoftBox,))(GumbelVolumeRatio)
register_measure("point-to-region distance")(PointToRegion)
register_measure("octagon bounds", default_for=(OctagonProduct,))(OctagonBounds)
register_measure("prism bounds", default_for=(Prism,))(PrismBounds)
register_measure("ellipsoid", default_for=(Ellipsoid,))(EllipsoidMeasure)
register_measure("cosine order", default_for=(Vector,))(CosineOrder)
register_measure("point distance", default_for=(Point,))(PointDistance)
