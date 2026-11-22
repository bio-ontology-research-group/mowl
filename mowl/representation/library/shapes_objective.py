r"""The objective of the shapes group: :class:`~mowl.representation.Objective` with the
options the four methods need, none of which changes its default reading.

Options added to the core objective:

``disjoint_margin``
    Margin of :math:`C \sqcap D \sqsubseteq \bot` when it differs from ``margin``
    (``None``: the same).
``neg_margins``
    Margin of the negative (apart, outside, shrunk) reading per axiom form, where it differs
    from ``margin``: ``{"gci2": 0.1, "class_assertion": 0.15}``.
``gci2_negative``
    The core options and, additionally, ``"outside"`` (the region of :math:`C` sticks out of
    that of :math:`\exists r.D`: ``measure.outside``) and ``"shrunk"`` (the published
    negative loss of the prism module: ``measure.meets_shrunk``).
``empty_margin``
    Pass ``margin`` to ``measure.empty`` (the prism module's loss for
    :math:`C \sqsubseteq \bot` contains it).
``weights``
    A factor per axiom form (Concept2Box weights its two views).
``ranking``
    ``{form: gamma}``: a negative cost is the positive cost minus ``gamma``, so that the
    ranking loss :math:`[\gamma + f(\delta) - f(\delta')]_+` of EIKE is
    ``relu(positive - negative)``.
``assertion``
    ``"region"`` (default): :math:`r(a, b)` is :math:`a \in \exists r.\{b\}`. ``"distance"``:
    the Euclidean distance of the image of the point :math:`a` to the point :math:`b` (the
    octagon module's loss), which differs from the region reading in how a deviation is
    weighed; negatives are not defined.
``context_axioms``
    Read :math:`C \sqsubseteq D` and :math:`C \sqsubseteq \exists r.D` as the triples of
    Concept2Box: the boxes of the classes in the context of the relation (``subclass`` for
    :math:`C \sqsubseteq D`), and for the second the box of :math:`D` inside that of :math:`C`
    (the direction of the published loss, reversed with respect to the axiom).
"""

import torch as th

from mowl.representation.library.shapes_measures import member_cost
from mowl.representation.objectives import Objective

__all__ = ["ShapeObjective"]


class ShapeObjective(Objective):
    """The objective of the shapes group; see the module docstring for the options."""

    def __init__(self, margin=0.0, slack=False, epsilon=0.01, gci3="inclusion",
                 gci2_negative="apart", negatives="all", empty="size", gci2_offset=0.0,
                 disjoint_margin=None, neg_margins=None, empty_margin=False, weights=None,
                 ranking=None, assertion="region", context_axioms=False):
        extra = ("outside", "shrunk")
        super().__init__(margin, slack, epsilon, gci3,
                         "apart" if gci2_negative in extra else gci2_negative, negatives,
                         empty, gci2_offset)
        if assertion not in ("region", "distance"):
            raise ValueError("assertion must be 'region' or 'distance'.")
        self.gci2_negative = gci2_negative
        self.disjoint_margin = disjoint_margin
        self.neg_margins = dict(neg_margins or {})
        self.empty_margin = bool(empty_margin)
        self.weights = dict(weights or {})
        self.ranking = None if ranking is None else dict(ranking)
        self.assertion = assertion
        self.context_axioms = bool(context_axioms)

    def _neg_margin(self, form):
        return self.neg_margins.get(form, self.margin)

    # -- weights and the ranking form ------------------------------------------------------
    def cost(self, g, form, rows, neg=False):
        if self.ranking is not None and neg and form in self.ranking:
            return super().cost(g, form, rows, False) - self.ranking[form]
        cost = super().cost(g, form, rows, neg)
        return cost * self.weights[form] if form in self.weights else cost

    # -- forms ----------------------------------------------------------------------------
    def gci0(self, g, rows, neg=False):
        if not self.context_axioms:
            return super().gci0(g, rows, neg)
        cm = g.class_model
        a = cm.in_context(rows[:, 0], cm.subclass_id)
        b = cm.in_context(rows[:, 1], cm.subclass_id)
        if self._neg(neg):
            return g.measure.apart(a, b, self.margin)
        return g.measure.inside(a, b, self.m)

    def gci2(self, g, rows, neg=False):
        if not self.context_axioms:
            return super().gci2(g, rows, neg)
        cm = g.class_model
        a = cm.in_context(rows[:, 2], rows[:, 1])
        b = cm.in_context(rows[:, 0], rows[:, 1])
        if neg:
            return g.measure.apart(a, b, self.margin)
        return g.measure.inside(a, b, self.m)

    def _existential(self, g, sub, relation, filler, neg, form="gci2"):
        mode = self.gci2_negative
        if not neg or mode in ("linear",):
            return super()._existential(g, sub, relation, filler, neg)
        target = g.exists(relation, filler)
        margin = self._neg_margin(form)
        if mode == "outside":
            return g.measure.outside(sub, target, margin)
        if mode == "shrunk":
            return g.measure.meets_shrunk(sub, target, -margin)
        if mode == "overlap":
            return g.measure.meets(sub, target, -margin)
        return g.measure.apart(sub, target, margin)

    def object_property_assertion(self, g, rows, neg=False):
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        if self.assertion == "distance":
            image = g.relation_model.image(rows[:, 1], a)
            return th.linalg.norm(image.centre - b.centre, dim=1, keepdim=True)
        return self._existential(g, a, rows[:, 1], b, neg, "object_property_assertion")

    def class_assertion(self, g, rows, neg=False):
        a, c = g.individuals(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg):
            return g.measure.apart(a, c, self._neg_margin("class_assertion"))
        return member_cost(g.measure, a, c, self.m)

    def gci1_bot(self, g, rows, neg=False):
        if self._neg(neg) or self.disjoint_margin is None:
            return super().gci1_bot(g, rows, neg)
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        return g.measure.apart(c, d, self.disjoint_margin)

    def _bottom(self, g, c, neg):
        if self.empty_margin and not self._neg(neg):
            return g.measure.empty(c, self.margin)
        return super()._bottom(g, c, neg)

    def gci3_bot(self, g, rows, neg=False):
        return self._bottom(g, g.classes(rows[:, 1 if rows.shape[1] > 1 else 0]), neg)
