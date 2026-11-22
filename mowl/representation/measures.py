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

from mowl.representation.regions import Ball, Box, ConstructorUnavailable, CornerBox, Region

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
        for region in args:
            if not isinstance(region, Region):
                continue
            if accepts and not isinstance(region, accepts):
                raise ConstructorUnavailable(
                    f"{type(self).__name__}.{method.__name__} compares "
                    f"{[c.__name__ for c in accepts]} regions, not "
                    f"{type(region).__name__} (a combination of parts that cannot "
                    f"read this axiom form).")
            reason = self.refuses(region)
            if reason:
                raise ConstructorUnavailable(f"{type(self).__name__}.{method.__name__}: "
                                             f"{reason}")
        return method(self, *args, **kwargs)
    relation.checked = True
    return relation


class RegionMeasure:
    """Base class of measures.

    :cvar accepts: The region classes the measure compares.
    """

    accepts = ()
    #: Vocabulary entry (:class:`mowl.representation.Measure` value) of the measure.
    part = ""
    #: What a zero cost means. ``"truth"``: the relation holds (exactly, or as a sufficient
    #: condition where a docstring says so); ``"score"``: the cost is a ranking score of a
    #: published method, which has no truth condition (zero says nothing, and the cost
    #: may not even reach zero where the relation holds). The soundness tests check the
    #: former and require the latter to be declared.
    reading = "truth"

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
                raise ConstructorUnavailable(
                    f"{type(self).__name__} compares "
                    f"{[c.__name__ for c in self.accepts]} regions, not "
                    f"{type(r).__name__}.")

    def inside(self, a, b, margin=0.0):
        """Zero iff ``a`` lies inside ``b`` (with ``margin`` room)."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} has no reading of 'inside' (it does not compare regions "
            f"in that way).")

    def meets(self, a, b, margin=0.0):
        """Zero iff ``a`` and ``b`` overlap (by at least ``margin``)."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} has no reading of 'meets' (it does not compare regions "
            f"in that way).")

    def apart(self, a, b, margin=0.0):
        """Zero iff ``a`` and ``b`` are disjoint (at least ``margin`` apart)."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} has no reading of 'apart' (it does not compare regions "
            f"in that way).")

    def meets_all(self, a, b, margin=0.0):
        """``meets`` for the negative reading "overlap": every part of a region made of
        several parts counts (a union of boxes). Default: ``meets``."""
        return self.meets(a, b, margin)

    def empty(self, a, margin=0.0):
        """Zero iff ``a`` is empty (has no extent, with ``margin`` room)."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} has no reading of 'empty' (it does not compare regions "
            f"in that way).")

    def nonempty(self, a, epsilon=0.0):
        """Zero iff ``a`` has extent at least ``epsilon``."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} has no reading of 'nonempty' (it does not compare regions "
            f"in that way).")

    def member(self, a, b, margin=0.0):
        """Zero iff the individual ``a`` is a member of ``b``. Default: ``inside``; a
        measure overrides it where an individual is compared differently from a class
        (a point against a region)."""
        return self.inside(a, b, margin)

    def outside(self, a, b, margin=0.0):
        """Zero iff ``a`` is *not* inside ``b``, by at least ``margin`` (non-inclusion; the
        negative term of RegD). Not every measure has it."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no non-inclusion.")

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
