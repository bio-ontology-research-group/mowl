"""Objectives: how each axiom form is read as a relation between regions.

An :class:`Objective` needs nothing but a geometry
(:class:`mowl.representation.Geometry`): it asks for the regions of the classes of an
axiom, builds the compound ones (:math:`\\exists r.C`, :math:`C \\sqcap D`) and has the
measure compare them. The default reading is the semantic one:

=============================  ==============================================
:math:`C \\sqsubseteq D`         ``inside(C, D)``
:math:`C \\sqcap D \\sqsubseteq E`  ``inside(C and D, E)``
:math:`C \\sqsubseteq \\exists r.D`  ``inside(C, exists r.D)``
:math:`\\exists r.C \\sqsubseteq D`  ``inside(exists r.C, D)``
:math:`C \\sqsubseteq \\bot`       ``empty(C)``
:math:`C \\sqcap D \\sqsubseteq \\bot`  ``apart(C, D)``
:math:`C(a)`                    ``inside(a, C)``
:math:`r(a, b)`                 ``inside(a, exists r.{b})``
=============================  ==============================================

A negative axiom (``neg=True``) asks for the two regions to be apart, as in DELE. The
options reproduce what publications and their codes do differently; the constructors
:meth:`Objective.elem` and :meth:`Objective.elbe` set them as in those methods.
"""

import torch as th

__all__ = ["Objective", "FORMS"]

#: The axiom forms an objective reads.
FORMS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
         "class_assertion", "object_property_assertion")


class Objective:
    """The reading of the axiom forms.

    :param margin: The margin :math:`\\gamma \\ge 0`.
    :param slack: ``False`` (default): the margin is strict, relations have to hold with
        :math:`\\gamma` to spare and a zero loss implies the axiom. ``True``: the margin is
        a slack, as in ELEmbeddings.
    :param epsilon: Smallest extent of a class that must not be empty.
    :param gci3: ``"inclusion"`` (default): :math:`\\exists r.C` lies inside :math:`D`.
        ``"overlap"``: it only has to overlap :math:`D`, as the published losses of
        ELEmbeddings and ELBE have it.
    :param gci2_negative: ``"apart"`` (default), ``"overlap"`` (the negative loss of ELBE:
        zero as soon as the two regions overlap by less than the margin) or ``"linear"``
        (``apart`` without its hinge, the ELEm baseline of the Box2EL code).
    :param negatives: ``"all"`` (default): every form has a negative loss. ``"gci2"``:
        only :math:`C \\sqsubseteq \\exists r.D` and role assertions have one; for the
        other forms ``neg=True`` returns the positive loss, as the original modules do.
    :param empty: ``"size"`` (default): the loss of :math:`C \\sqsubseteq \\bot` is the
        size of the region. ``"raw"``: the signed radius parameter, as in the ELEmbeddings
        code (unbounded below).
    :param gci2_offset: Constant added to the positive loss of
        :math:`C \\sqsubseteq \\exists r.D` (``1e-5`` in the ELEmbeddings code).
    """

    def __init__(self, margin=0.0, slack=False, epsilon=0.01, gci3="inclusion",
                 gci2_negative="apart", negatives="all", empty="size", gci2_offset=0.0):
        if gci3 not in ("inclusion", "overlap"):
            raise ValueError("gci3 must be 'inclusion' or 'overlap'.")
        if gci2_negative not in ("apart", "overlap", "linear"):
            raise ValueError("gci2_negative must be 'apart', 'overlap' or 'linear'.")
        if negatives not in ("all", "gci2"):
            raise ValueError("negatives must be 'all' or 'gci2'.")
        if empty not in ("size", "raw"):
            raise ValueError("empty must be 'size' or 'raw'.")
        self.margin = float(margin)
        self.slack = bool(slack)
        self.epsilon = float(epsilon)
        self.gci3_reading = gci3
        self.gci2_negative = gci2_negative
        self.negatives = negatives
        self.empty = empty
        self.gci2_offset = float(gci2_offset)

    # -- the readings of publications -----------------------------------------------------
    @classmethod
    def elem(cls, margin=0.1, all_negatives=False, linear_negative=False, epsilon=0.01,
             gci3="overlap"):
        """ELEmbeddings (Kulmanov et al. 2019) as implemented in mOWL."""
        return cls(margin, slack=True, epsilon=epsilon, gci3=gci3,
                   gci2_negative="linear" if linear_negative else "apart",
                   negatives="all" if all_negatives else "gci2", empty="raw",
                   gci2_offset=10e-6)

    @classmethod
    def elbe(cls, margin=0.1, all_negatives=False, epsilon=0.01, gci3="overlap"):
        """ELBE (Peng et al. 2022) as implemented in mOWL."""
        return cls(margin, slack=False, epsilon=epsilon, gci3=gci3,
                   gci2_negative="overlap", negatives="all" if all_negatives else "gci2")

    # -- margins ----------------------------------------------------------------------------
    @property
    def m(self):
        """Margin of ``inside`` and ``meets``: positive when strict, negative as a slack."""
        return -self.margin if self.slack else self.margin

    def _neg(self, neg):
        return neg and self.negatives == "all"

    # -- the forms --------------------------------------------------------------------------
    def cost(self, g, form, rows, neg=False):
        """The loss of the axioms ``rows`` of form ``form`` in the geometry ``g``,
        ``(B, 1)``."""
        if form not in FORMS:
            raise ValueError(f"Unknown axiom form {form!r}; known: {FORMS}.")
        return getattr(self, form)(g, rows, neg)

    def gci0(self, g, rows, neg=False):
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg):
            return g.measure.apart(c, d, self.margin)
        return g.measure.inside(c, d, self.m)

    def gci1(self, g, rows, neg=False):
        c, d, e = (g.classes(rows[:, i]) for i in range(3))
        if self._neg(neg):
            return g.measure.intersection_apart(c, d, e, self.margin)
        return g.measure.intersection_inside(c, d, e, self.m)

    def _existential(self, g, sub, relation, filler, neg):
        """``sub`` against :math:`\\exists r.\\text{filler}`."""
        target = g.exists(relation, filler)
        if not neg:
            return g.measure.inside(sub, target, self.m) + self.gci2_offset
        if self.gci2_negative == "overlap":
            return g.measure.meets(sub, target, -self.margin)
        if self.gci2_negative == "linear":
            return _unhinged_apart(g.measure, sub, target, self.margin)
        return g.measure.apart(sub, target, self.margin)

    def gci2(self, g, rows, neg=False):
        return self._existential(g, g.classes(rows[:, 0]), rows[:, 1],
                                 g.classes(rows[:, 2]), neg)

    def gci3(self, g, rows, neg=False):
        exists = g.exists(rows[:, 0], g.classes(rows[:, 1]))
        d = g.classes(rows[:, 2])
        if self._neg(neg):
            return g.measure.apart(exists, d, self.margin)
        if self.gci3_reading == "overlap":
            return g.measure.meets(exists, d, self.m)
        return g.measure.inside(exists, d, self.m)

    def _bottom(self, g, c, neg):
        if self._neg(neg):
            return g.measure.nonempty(c, self.epsilon)
        if self.empty == "raw":
            if getattr(c, "raw", None) is None:
                raise TypeError("empty='raw' needs a class model that keeps a signed size.")
            return c.raw
        return g.measure.empty(c)

    def gci0_bot(self, g, rows, neg=False):
        return self._bottom(g, g.classes(rows[:, 0]), neg)

    def gci3_bot(self, g, rows, neg=False):
        # exists r.C is empty iff C is.
        return self._bottom(g, g.classes(rows[:, 1]), neg)

    def gci1_bot(self, g, rows, neg=False):
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg):
            return g.measure.intersection_nonempty(c, d, -self.margin, self.epsilon)
        return g.measure.apart(c, d, self.margin)

    def class_assertion(self, g, rows, neg=False):
        a, c = g.individuals(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg):
            return g.measure.apart(a, c, self.margin)
        return g.measure.inside(a, c, self.m)

    def object_property_assertion(self, g, rows, neg=False):
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        target = g.exists(rows[:, 1], b)
        if not neg:
            return g.measure.inside(a, target, self.m) + self.gci2_offset
        if self.gci2_negative == "overlap":
            return g.measure.meets(a, target, -self.margin)
        return g.measure.apart(a, target, self.margin)


def _unhinged_apart(measure, a, b, margin):
    """``apart`` without the hinge, for ball regions (the only published use)."""
    distance = th.linalg.norm(a.centre - b.centre, dim=1, keepdim=True)
    return a.radius + b.radius - distance + margin
