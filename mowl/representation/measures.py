"""Measures: how two regions are compared.

A measure turns a relation between regions into a cost per row (``(B, 1)``, never
negative, lower is better): :meth:`~RegionMeasure.inside` (:math:`A \\subseteq B`),
:meth:`~RegionMeasure.meets` (:math:`A \\cap B \\neq \\emptyset`), :meth:`~RegionMeasure.apart`
(:math:`A \\cap B = \\emptyset`), :meth:`~RegionMeasure.empty` and :meth:`~RegionMeasure.nonempty`.
Three more are read by the objectives where a method needs them: :meth:`~RegionMeasure.member`
(an individual lies in a class), :meth:`~RegionMeasure.outside` (a region sticks out of
another) and :meth:`~RegionMeasure.negative` (the cost of a pair that should *not* be
included, which is where the measures of the hierarchy methods differ most).

Every relation takes a ``margin``. A positive margin is *strict*: the relation has to hold
with that much room, and a zero cost implies the relation. A negative margin is a *slack*:
the relation may be violated by that much at zero cost. Which of the two a publication
means by its :math:`\\gamma` differs (ELEmbeddings: slack; ELBE: strict), see
:mod:`mowl.training.margin`.
"""

import functools

import torch as th

from mowl.representation.operations import Operations, generic
from mowl.representation.regions import (Ball, Box, ConstructorNotImplemented,
                                         ConstructorUnavailable, CornerBox,
                                         MarginNotImplemented, MarginUnavailable, Region,
                                         family_refusal)

__all__ = ["RegionMeasure", "CentreDistance", "Bounds", "lens_balls"]

#: The relations of a measure whose region arguments are checked against ``accepts``.
RELATIONS = ("inside", "meets", "apart", "meets_all", "empty", "nonempty", "member",
             "outside", "negative", "intersection_inside", "intersection_apart",
             "intersection_nonempty")


def _checked(method):
    """``method`` with its region arguments checked against the ``accepts`` of the measure:
    a region of another family is an error that says so
    (:class:`~mowl.representation.ConstructorUnavailable`), not whatever the formula does
    with tensors it was not written for."""
    @functools.wraps(method)
    def relation(self, *args, **kwargs):
        accepts = self.accepts
        if accepts and any(isinstance(r, Region) and not isinstance(r, accepts)
                           and type(r).viewable_as(accepts) for r in args):
            # a region that is exactly one of the accepted family (a point is a box of no
            # width) is compared as that region
            args = tuple(r.view_as(accepts) if isinstance(r, Region) and not isinstance(
                r, accepts) and type(r).viewable_as(accepts) else r for r in args)
        for region in args:
            if not isinstance(region, Region):
                continue
            if accepts and not isinstance(region, accepts):
                raise family_refusal(f"{type(self).__name__}.{method.__name__}", accepts,
                                     region)
            reason = self.refuses(region)
            if reason:
                raise ConstructorUnavailable(f"{type(self).__name__}.{method.__name__}: "
                                             f"{reason}")
        return method(self, *args, **kwargs)
    relation.checked = True
    return relation


class RegionMeasure(Operations):
    """Base class of measures.

    :cvar accepts: The region classes the measure compares.

    Besides the relations every measure has, a measure may provide the optional operations
    of :attr:`OPERATIONS` (:mod:`mowl.representation.operations`); an objective asks for
    them with :meth:`~mowl.representation.operations.Operations.operation`. Those with a
    definition in terms of the common interface have it here (``negative``,
    ``product_overlap``, ``meets_shrunk``, ``is_empty``), so that every measure has them;
    a measure overrides one where its own formula is exact or is the published one.
    """

    OPERATIONS = {
        "outside": "zero iff a is not inside b, by at least the margin (non-inclusion); "
                   "generic: the centre of a is apart from b (a sufficient condition)",
        "negative": "the cost of a pair that should not be included; generic: apart",
        "product_overlap": "Vol(A n B) / (Vol(A) Vol(B)), the BoxEL code's reading of "
                           "disjointness; generic: the measure's intersection and the "
                           "regions' volumes",
        "meets_shrunk": "zero when a, contracted towards its centre, meets b (the published "
                        "negative of the prism module); generic: meets(a contracted to its "
                        "centre, b)",
        "is_empty": "(B, 1) boolean: the region has no point; generic: the region's own "
                    "is_empty",
        "host_for": "the measure that reads the relations this one does not define, for a "
                    "region family (RegD's energy defines inclusion only); generic: the "
                    "measure itself",
    }

    accepts = ()
    #: Vocabulary entry (:class:`mowl.representation.Measure` value) of the measure.
    part = ""
    #: What a zero cost means. ``"truth"``: the relation holds (exactly, or as a sufficient
    #: condition where a docstring says so); ``"score"``: the cost is a ranking score of a
    #: published method, which has no truth condition (zero says nothing, and the cost
    #: may not even reach zero where the relation holds). The soundness tests check the
    #: former and require the latter to be declared.
    reading = "truth"

    #: Whether ``meets`` is the distance of the two regions (zero iff they overlap, otherwise
    #: how far apart they are): the Euclidean distance of two boxes or of two balls. The
    #: negative of an existential reads such a measure in the published form of Box2EL and
    #: DELE; for any other measure it is a hinge
    #: (:meth:`~mowl.representation.Relations.subsumed_negative`).
    meets_is_distance = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in RELATIONS:
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "checked", False):
                setattr(cls, name, _checked(method))

    def refuses(self, region):
        """Why the measure cannot compare ``region`` although it is of a family it accepts
        (a fuzzy set of logits for a measure of degrees), or ``None``."""
        return None

    def check(self, *regions):
        """Raise :class:`~mowl.representation.ConstructorUnavailable` (a
        :class:`TypeError`) if a region is not of a family the measure compares."""
        for r in regions:
            if not isinstance(r, self.accepts):
                raise family_refusal(type(self).__name__, self.accepts, r)

    def inside(self, a, b, margin=0.0):
        """Zero iff ``a`` lies inside ``b`` (with ``margin`` room)."""
        raise ConstructorNotImplemented(
            f"{type(self).__name__} has no formula for 'inside' of its regions.")

    def meets(self, a, b, margin=0.0):
        """Zero iff ``a`` and ``b`` overlap (by at least ``margin``)."""
        raise ConstructorNotImplemented(
            f"{type(self).__name__} has no formula for 'meets' of its regions.")

    def apart(self, a, b, margin=0.0):
        """Zero iff ``a`` and ``b`` are disjoint (at least ``margin`` apart)."""
        raise ConstructorNotImplemented(
            f"{type(self).__name__} has no formula for 'apart' of its regions.")

    def meets_all(self, a, b, margin=0.0):
        """``meets`` for the negative reading "overlap": every part of a region made of
        several parts counts (a union of boxes). Default: ``meets``."""
        return self.meets(a, b, margin)

    def empty(self, a, margin=0.0):
        """Zero iff ``a`` is empty (has no extent, with ``margin`` room)."""
        raise ConstructorNotImplemented(
            f"{type(self).__name__} has no formula for 'empty' of its regions.")

    def nonempty(self, a, epsilon=0.0):
        """Zero iff ``a`` has extent at least ``epsilon``."""
        raise ConstructorNotImplemented(
            f"{type(self).__name__} has no formula for 'nonempty' of its regions.")

    def member(self, a, b, margin=0.0):
        """Zero iff the individual ``a`` is a member of ``b``. Default: ``inside``; a
        measure overrides it where an individual is compared differently from a class
        (a point against a region)."""
        return self.inside(a, b, margin)

    @generic
    def outside(self, a, b, margin=0.0):
        """Zero only if ``a`` is *not* inside ``b``, by at least ``margin`` (non-inclusion;
        the negative term of RegD). Generic definition, a sufficient condition: the centre of
        ``a`` (``a`` contracted to its centre, :meth:`Region.contract
        <mowl.representation.Region.contract>`) is :meth:`apart` from ``b`` by ``margin``, so
        a point of ``a`` is outside ``b``. A measure whose own formula is the exact test of
        its family (boxes, balls, octagons) overrides it. A region family that cannot be
        contracted, or a measure without ``apart``, refuses."""
        return self.apart(a.contract(), b, margin)

    @generic
    def negative(self, a, b, margin=0.0):
        """The cost of a pair ``a``, ``b`` that should *not* be included. Default:
        ``apart`` (the semantic reading, which DELE uses); measures of methods that read a
        negative pair differently (a log-probability, a hinge on the energy, a
        non-inclusion) override it."""
        return self.apart(a, b, margin)

    def intersect(self, a, b):
        """``(region, gap)`` of the conjunction of the regions ``a`` and ``b`` under this
        measure. Default: the regions intersect themselves (families closed under
        intersection); a measure on membership functions forms the minimum instead."""
        return a.intersect(b)

    @generic
    def product_overlap(self, a, b, eps=1e-12):
        """:math:`\\mathrm{Vol}(A \\cap B) / (\\mathrm{Vol}(A)\\,\\mathrm{Vol}(B))`, ``(B,
        1)``, from the exact intersection (:meth:`intersect`) and the volumes of the
        regions (:meth:`Region.volume <mowl.representation.Region.volume>`): zero iff the
        two regions do not overlap in a set of positive volume. A region family without an
        intersection in the family or without a volume refuses."""
        region, _ = self.intersect(a, b)
        return region.volume() / (a.volume() * b.volume()).clamp_min(eps)

    @generic
    def meets_shrunk(self, a, b, margin=0.0):
        """Zero when ``a`` contracted to its centre (:meth:`Region.contract
        <mowl.representation.Region.contract>`) meets ``b`` (with ``margin``): the centre of
        ``a`` lies in ``b``, which is stronger than an overlap."""
        return self.meets(a.contract(), b, margin)

    @generic
    def host_for(self, region_class):
        """The measure of the relations this measure does not define, for regions of
        ``region_class``: a measure that defines them all is its own host."""
        return self

    @generic
    def is_empty(self, a):
        """``(B, 1)`` boolean: the region has no point (:meth:`Region.is_empty
        <mowl.representation.Region.is_empty>`)."""
        return a.is_empty().reshape(-1, 1)

    @staticmethod
    def grown(a, margin, what):
        """The region ``a`` enlarged by a strict ``margin`` (the Minkowski sum with a ball or
        box of that radius), for a measure whose formula has no margin of its own: the
        relation holds with room ``margin`` iff it holds for the enlarged region, which is
        how :class:`Bounds` and :class:`CentreDistance` read a strict margin. A slack (a
        negative margin) is refused (:class:`~mowl.representation.MarginUnavailable`):
        shrinking a region makes it empty, which such a measure does not read as the empty
        set."""
        if not margin:
            return a
        if margin < 0:
            raise MarginNotImplemented(
                f"{what} reads a strict margin as the enlarged region, and the reading of a "
                f"slack (margin {margin!r}) as the shrunk region is not written.")
        if not isinstance(a, Region):
            raise MarginUnavailable(f"{what} reads a strict margin as the enlarged region; "
                                    f"a {type(a).__name__} cannot be enlarged.")
        try:
            return a.grow(margin)
        except ConstructorNotImplemented as error:
            raise MarginNotImplemented(
                f"{what} reads a strict margin as the enlarged region: "
                f"{error.reworded()}") from error
        except ConstructorUnavailable as error:
            raise MarginUnavailable(f"{what} reads a strict margin as the enlarged region: "
                                    f"{error}") from error

    def bind(self, relations):
        """Called with the relation model by a geometry that is built from this measure,
        for measures whose cost reads the relation model (a triple score). Default: nothing
        to bind. Returns the measure."""
        return self

    # -- relations of an intersection: exact where the regions are closed under it --------
    #: Whether the three methods below compute the intersection exactly.
    exact_intersection = True

    def intersection_inside(self, a, b, c, margin=0.0):
        """Zero iff :math:`a \\cap b` is not empty and lies inside ``c``."""
        region, gap = a.intersect(b)
        return self.inside(region, c, margin) + gap

    def intersection_apart(self, a, b, c, margin=0.0):
        """Zero iff :math:`a \\cap b` is not empty and disjoint from ``c``."""
        region, gap = a.intersect(b)
        return self.apart(region, c, margin) + gap

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        """Zero iff :math:`a \\cap b` has extent at least ``epsilon``."""
        region, _ = a.intersect(b)
        return self.nonempty(region, epsilon)


def lens_balls(a, b, eps=1e-12):
    """The smallest ball that contains the intersection (the *lens*) of two balls, and the
    largest ball on their axis that it contains: ``(outer, inner, disjoint)``, with
    ``disjoint`` ``(B, 1)`` true where the balls do not meet.

    With :math:`d` the distance of the centres and :math:`x = (d^2 + r_a^2 - r_b^2) / 2d`
    the distance of the plane of the intersection circle from :math:`c_a`: if one ball lies
    in the other, both are the smaller ball; if :math:`0 \\le x \\le d` the outer ball is
    the ball of the intersection circle (centre :math:`c_a + x u`, radius
    :math:`\\sqrt{r_a^2 - x^2}`), else the lens contains a great circle of the smaller ball,
    which is then the outer ball. The inner ball is centred half way between the two ends
    of the lens on the axis, with half its thickness :math:`(r_a + r_b - d) / 2` as radius.
    """
    ca, cb, ra, rb = a.centre, b.centre, a.radius, b.radius
    diff = cb - ca
    d = th.linalg.norm(diff, dim=-1, keepdim=True)
    safe = d.clamp_min(eps)
    unit = diff / safe
    nested = d <= th.abs(ra - rb)
    smaller_centre = th.where(ra <= rb, ca, cb)
    smaller_radius = th.minimum(ra, rb)
    x = (d ** 2 + ra ** 2 - rb ** 2) / (2 * safe)
    circle = (~nested) & (x >= 0) & (x <= d)
    h = th.sqrt(th.relu(ra ** 2 - x ** 2) + eps ** 2)
    outer = Ball(th.where(circle, ca + x * unit, smaller_centre),
                 th.where(circle, h, smaller_radius))
    middle = (d - rb + ra) / 2
    inner = Ball(th.where(nested, smaller_centre, ca + middle * unit),
                 th.where(nested, smaller_radius, th.relu(ra + rb - d) / 2))
    return outer, inner, d >= ra + rb


class CentreDistance(RegionMeasure):
    """Balls compared by the distance of their centres against their radii:
    :math:`A \\subseteq B` iff :math:`\\lVert c_A - c_B \\rVert + r_A \\le r_B`.

    Two balls do not intersect in a ball, so the forms with a conjunction are read through
    balls that bound the intersection (:func:`lens_balls`):

    :param conjunction: ``"bounding"`` (default): :math:`A \\sqcap B \\sqsubseteq C` costs
        nothing where the balls do not meet, otherwise the inclusion of the smallest ball
        around the intersection in :math:`C` (a sufficient condition: a zero cost implies the
        axiom); its negative is the overlap of :math:`A` and :math:`B` and the disjointness of
        the largest ball inside their intersection from :math:`C` (also sufficient).
        ``"published"``: the losses of ELEmbeddings (Kulmanov et al. 2019, Eq. 2: the two
        balls overlap, the centre of :math:`C` lies in both, and :math:`C` is at least as
        large as the smaller of the two) and of DELE (Mashkova et al. 2024, Eq. 3); a zero
        loss does *not* imply the axiom (a ball centred in the intersection, as large as the
        smaller ball, need not contain it).
    :param radius_term: With ``"published"``: whether the conjunction loss has its fourth
        term, the radius of :math:`C` against the smaller radius, which the authors' code
        leaves out.

    Part (vocabulary value): ``centre distance``. Used by: ``elem``, ``emelpp``, ``emelvar``.
    """

    meets_is_distance = True

    accepts = (Ball,)
    part = "centre distance"
    exact_intersection = False

    def __init__(self, radius_term=True, conjunction="bounding"):
        if conjunction not in ("bounding", "published"):
            raise ValueError("conjunction must be 'bounding' or 'published'.")
        #: Whether :meth:`intersection_inside` has the fourth term of the published
        #: ELEmbeddings loss, which the authors' code leaves out.
        self.radius_term = bool(radius_term)
        self.conjunction = conjunction

    def intersection_inside(self, a, b, c, margin=0.0):
        """Zero (``"bounding"``) only if :math:`A \\cap B \\subseteq C`; see the class."""
        self.check(a, b, c)
        if self.conjunction == "bounding":
            outer, _, disjoint = lens_balls(a, b)
            cost = self.inside(outer, c, margin)
            return th.where(disjoint, th.zeros_like(cost), cost)
        cost = (self.meets(a, b, margin)
                + th.relu(self._distance(c, a) - a.radius + margin)
                + th.relu(self._distance(c, b) - b.radius + margin))
        if self.radius_term:
            cost = cost + th.relu(th.minimum(a.radius, b.radius) - c.radius + margin)
        return cost

    def intersection_apart(self, a, b, c, margin=0.0):
        """Zero (``"bounding"``) only if :math:`A \\cap B` is not empty and has points
        outside :math:`C`; ``"published"``: DELE, the two balls overlap (with slack) and the
        centre of :math:`C` lies outside both."""
        self.check(a, b, c)
        if self.conjunction == "bounding":
            _, inner, _ = lens_balls(a, b)
            return self.meets(a, b, margin) + self.apart(inner, c, margin)
        return (self.meets(a, b, -margin)
                + th.relu(a.radius - self._distance(a, c) + margin)
                + th.relu(b.radius - self._distance(b, c) + margin))

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        self.check(a, b)
        return self.meets(a, b, margin)

    @staticmethod
    def _distance(a, b):
        return th.linalg.norm(a.centre - b.centre, dim=1, keepdim=True)

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        return th.relu(self._distance(a, b) + a.radius - b.radius + margin)

    def meets(self, a, b, margin=0.0):
        self.check(a, b)
        return th.relu(self._distance(a, b) - a.radius - b.radius + margin)

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        return th.relu(a.radius + b.radius - self._distance(a, b) + margin)

    def empty(self, a, margin=0.0):
        self.check(a)
        return a.radius + margin if margin else a.radius

    def nonempty(self, a, epsilon=0.0):
        self.check(a)
        return th.relu(epsilon - a.radius)

    def outside(self, a, b, margin=0.0):
        """Zero iff ``a`` sticks out of ``b`` by at least ``margin``:
        :math:`\\lVert c_A - c_B \\rVert + r_A - r_B \\ge m`."""
        self.check(a, b)
        return th.relu(b.radius - a.radius - self._distance(a, b) + margin)


class Bounds(RegionMeasure):
    """Boxes compared coordinate by coordinate: :math:`A \\subseteq B` iff
    :math:`|c_A - c_B| + o_A \\le o_B` in every coordinate. The cost is the norm of the
    violations.

    A box whose upper corner lies below its lower one in some coordinate (a negative
    half-width: the exact intersection of two boxes that are apart, the fixed bottom class
    of ELK-Em, the restriction of ELK-Em's relation to an empty part of its range) is the
    *empty* box: it lies inside every box, meets none and is apart from all.

    :param separation: When two boxes count as apart. ``"any"`` (default): they are
        separated in at least one coordinate, which is what disjointness of boxes means;
        the cost is the smallest overlap over the coordinates. ``"every"``: they are
        separated in every coordinate, as the losses of ELBE and of the codes derived from
        it require (a much stronger condition than disjointness).
    :param extent: How the size of a box is read by :meth:`empty` and :meth:`nonempty`.
        ``"thinnest"`` (default): by its smallest half-width; a box is empty (has no
        volume, contains no sampled point) as soon as one coordinate has no width, and has
        extent :math:`\\epsilon` when every coordinate has half-width :math:`\\epsilon`.
        ``"norm"``: by the norm of the half-widths (the losses of ELBE and of the methods
        derived from it): ``empty`` is then zero only for a point, and ``nonempty`` is zero
        for a flat box of no volume.
    :param conjunction: How :math:`A \\sqcap B \\sqsubseteq C` is read.
        ``"exact"`` (default): the exact intersection lies inside :math:`C` (zero where it is
        empty, since the empty set lies in every class); :math:`A \\sqcap B` having extent is
        read on the exact intersection as well. ``"nonempty"``: the intersection must also
        be non-empty, and an empty intersection is replaced by the box between the two
        (ELBE's loss: :math:`\\mathrm{inside}(|A \\cap B|, C) + \\mathrm{gap}`).
    :param empty_boxes: ``"empty"`` (default): an empty box is read as the empty set, as
        above. ``"formula"``: the formulas are applied to its negative half-widths as they
        are (the ELK-Em module: its fixed bottom class and the top class are then not
        apart by the margin).

    Part (vocabulary value): ``bounds per coordinate``. Used by: ``box2el``, ``elbe``,
    ``multiboxel``, ``multielbe``, ``query2box``.
    """

    meets_is_distance = True

    accepts = (Box,)
    part = "bounds per coordinate"

    def __init__(self, separation="any", extent="thinnest", conjunction="exact",
                 empty_boxes="empty"):
        if separation not in ("any", "every"):
            raise ValueError("separation must be 'any' or 'every'.")
        if extent not in ("thinnest", "norm"):
            raise ValueError("extent must be 'thinnest' or 'norm'.")
        if conjunction not in ("exact", "nonempty"):
            raise ValueError("conjunction must be 'exact' or 'nonempty'.")
        if empty_boxes not in ("empty", "formula"):
            raise ValueError("empty_boxes must be 'empty' or 'formula'.")
        self.separation = separation
        self.extent = extent
        self.conjunction = conjunction
        self.empty_boxes = empty_boxes

    @staticmethod
    def _norm(x):
        return th.linalg.norm(th.relu(x), dim=1, keepdim=True)

    @staticmethod
    def is_empty(a):
        """``(B, 1)`` boolean: the box has no point (upper below lower in some coordinate)."""
        return (a.upper < a.lower).any(dim=1, keepdim=True)

    def _empty(self, a):
        """:meth:`is_empty`, or never with ``empty_boxes="formula"``."""
        if self.empty_boxes == "formula":
            return th.zeros(len(a), 1, dtype=th.bool, device=a.centre.device)
        return self.is_empty(a)

    @staticmethod
    def _half(a):
        """Signed half-widths (negative where the box is empty in that coordinate)."""
        if isinstance(a, CornerBox):
            return (a.upper - a.lower) / 2
        return a.offset

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        cost = self._norm(th.abs(a.centre - b.centre) + a.offset - b.offset + margin)
        return th.where(self._empty(a), th.zeros_like(cost), cost)

    def meets(self, a, b, margin=0.0):
        self.check(a, b)
        cost = self._norm(th.abs(a.centre - b.centre) - a.offset - b.offset + margin)
        if self.empty_boxes == "formula":
            return cost
        # an empty box meets nothing (zero for boxes that are not empty)
        return cost + self._norm(a.lower - a.upper) + self._norm(b.lower - b.upper)

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        overlap = a.offset + b.offset - th.abs(a.centre - b.centre) + margin
        if self.separation == "every":
            cost = self._norm(overlap)
        else:
            cost = th.relu(overlap.min(dim=1, keepdim=True).values)
        return th.where(self._empty(a) | self._empty(b), th.zeros_like(cost), cost)

    def empty(self, a, margin=0.0):
        self.check(a)
        if self.extent == "norm":
            return th.linalg.norm(a.offset + margin if margin else a.offset, dim=1,
                                  keepdim=True)
        return th.relu(self._half(a).min(dim=1, keepdim=True).values + margin)

    def nonempty(self, a, epsilon=0.0):
        self.check(a)
        if self.extent == "norm":
            return th.relu(epsilon - th.linalg.norm(a.offset, dim=1, keepdim=True))
        return th.relu(epsilon - self._half(a).min(dim=1, keepdim=True).values)

    def intersect(self, a, b):
        """The exact intersection (an empty box where they are apart) and its gap."""
        self.check(a, b)
        return a.intersect(b, signed=True)

    def outside(self, a, b, margin=0.0):
        """Zero iff ``a`` sticks out of ``b`` by at least ``margin`` in some coordinate:
        :math:`\\max_i |c_{A,i} - c_{B,i}| + o_{A,i} - o_{B,i} \\ge m`. An empty ``a`` lies
        inside every box: its cost is never zero (the amount by which its upper corner lies
        below its lower one is added)."""
        self.check(a, b)
        stick_out = th.abs(a.centre - b.centre) + th.relu(self._half(a)) - b.offset
        cost = th.relu(margin - stick_out.max(dim=1, keepdim=True).values)
        return cost + self._norm(a.lower - a.upper)

    def intersection_inside(self, a, b, c, margin=0.0):
        """Zero iff :math:`A \\cap B \\subseteq C` (``conjunction="exact"``), or iff it
        is in addition not empty (``"nonempty"``)."""
        if self.conjunction == "exact":
            region, _ = a.intersect(b, signed=True)
            return self.inside(region, c, margin)
        region, gap = a.intersect(b, signed=False)
        return self.inside(region, c, margin) + gap

    def intersection_apart(self, a, b, c, margin=0.0):
        """Zero iff :math:`A \\cap B` is not empty and disjoint from :math:`C`."""
        region, gap = a.intersect(b, signed=False)
        return self.apart(region, c, margin) + gap

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        """Zero iff :math:`A \\cap B` has extent at least ``epsilon`` (with
        ``conjunction="nonempty"`` the box between two boxes that are apart is measured
        instead, as before)."""
        region, _ = a.intersect(b, signed=self.conjunction == "exact")
        return self.nonempty(region, epsilon)
