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

import torch as th

from mowl.representation.regions import Ball, Box, ConstructorUnavailable

__all__ = ["RegionMeasure", "CentreDistance", "Bounds"]


class RegionMeasure:
    """Base class of measures.

    :cvar accepts: The region classes the measure compares.
    """

    accepts = ()
    #: Vocabulary entry (:class:`mowl.representation.Measure` value) of the measure.
    part = ""

    def check(self, *regions):
        for r in regions:
            if not isinstance(r, self.accepts):
                raise TypeError(f"{type(self).__name__} compares "
                                f"{[c.__name__ for c in self.accepts]} regions, not "
                                f"{type(r).__name__}.")

    def inside(self, a, b, margin=0.0):
        """Zero iff ``a`` lies inside ``b`` (with ``margin`` room)."""
        raise NotImplementedError

    def meets(self, a, b, margin=0.0):
        """Zero iff ``a`` and ``b`` overlap (by at least ``margin``)."""
        raise NotImplementedError

    def apart(self, a, b, margin=0.0):
        """Zero iff ``a`` and ``b`` are disjoint (at least ``margin`` apart)."""
        raise NotImplementedError

    def meets_all(self, a, b, margin=0.0):
        """``meets`` for the negative reading "overlap": every part of a region made of
        several parts counts (a union of boxes). Default: ``meets``."""
        return self.meets(a, b, margin)

    def empty(self, a, margin=0.0):
        """Zero iff ``a`` is empty (has no extent, with ``margin`` room)."""
        raise NotImplementedError

    def nonempty(self, a, epsilon=0.0):
        """Zero iff ``a`` has extent at least ``epsilon``."""
        raise NotImplementedError

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


class CentreDistance(RegionMeasure):
    """Balls compared by the distance of their centres against their radii:
    :math:`A \\subseteq B` iff :math:`\\lVert c_A - c_B \\rVert + r_A \\le r_B`."""

    accepts = (Ball,)
    part = "centre distance"
    exact_intersection = False

    def __init__(self, radius_term=True):
        #: Whether :meth:`intersection_inside` has the fourth term of the published
        #: ELEmbeddings loss, which the authors' code leaves out.
        self.radius_term = bool(radius_term)

    def intersection_inside(self, a, b, c, margin=0.0):
        """Two balls do not intersect in a ball. The loss of ELEmbeddings (Kulmanov et al.
        2019, Eq. 2) is used instead: the two balls overlap, the centre of ``c`` lies in
        both, and ``c`` is at least as large as the smaller of the two."""
        self.check(a, b, c)
        cost = (self.meets(a, b, margin)
                + th.relu(self._distance(c, a) - a.radius + margin)
                + th.relu(self._distance(c, b) - b.radius + margin))
        if self.radius_term:
            cost = cost + th.relu(th.minimum(a.radius, b.radius) - c.radius + margin)
        return cost

    def intersection_apart(self, a, b, c, margin=0.0):
        """DELE (Mashkova et al. 2024, Eq. 3): the two balls overlap (with slack) and the
        centre of ``c`` lies outside both."""
        self.check(a, b, c)
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

    :param separation: When two boxes count as apart. ``"any"`` (default): they are
        separated in at least one coordinate, which is what disjointness of boxes means;
        the cost is the smallest overlap over the coordinates. ``"every"``: they are
        separated in every coordinate, as the losses of ELBE and of the codes derived from
        it require (a much stronger condition than disjointness).
    """

    accepts = (Box,)
    part = "bounds per coordinate"

    def __init__(self, separation="any"):
        if separation not in ("any", "every"):
            raise ValueError("separation must be 'any' or 'every'.")
        self.separation = separation

    @staticmethod
    def _norm(x):
        return th.linalg.norm(th.relu(x), dim=1, keepdim=True)

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        return self._norm(th.abs(a.centre - b.centre) + a.offset - b.offset + margin)

    def meets(self, a, b, margin=0.0):
        self.check(a, b)
        return self._norm(th.abs(a.centre - b.centre) - a.offset - b.offset + margin)

    def apart(self, a, b, margin=0.0):
        self.check(a, b)
        overlap = a.offset + b.offset - th.abs(a.centre - b.centre) + margin
        if self.separation == "every":
            return self._norm(overlap)
        return th.relu(overlap.min(dim=1, keepdim=True).values)

    def empty(self, a, margin=0.0):
        self.check(a)
        return th.linalg.norm(a.offset + margin if margin else a.offset, dim=1, keepdim=True)

    def nonempty(self, a, epsilon=0.0):
        self.check(a)
        return th.relu(epsilon - th.linalg.norm(a.offset, dim=1, keepdim=True))
