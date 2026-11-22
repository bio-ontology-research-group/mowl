r"""The reading of the shapes group that is its own: two readings of the axiom forms that belong
to one method each. Everything else the four methods need (``disjoint_margin``,
``neg_margins``, ``weights``, ``ranking``, ``empty_margin``, the ``"outside"`` and
``"shrunk"`` negatives, ``gci3_bot="filler"``) is an option of the core
:class:`~mowl.representation.Objective`.

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

from mowl.representation.objectives import Objective

__all__ = ["ShapeObjective"]


class ShapeObjective(Objective):
    """The core objective with the two readings above; the shapes methods read
    :math:`\\exists r.C \\sqsubseteq \\bot` through the filler (``gci3_bot="filler"``)."""

    def __init__(self, margin=0.0, *args, assertion="region", context_axioms=False, **kwargs):
        kwargs.setdefault("gci3_bot", "filler")
        super().__init__(margin, *args, **kwargs)
        if assertion not in ("region", "distance"):
            raise ValueError("assertion must be 'region' or 'distance'.")
        self.assertion = assertion
        self.context_axioms = bool(context_axioms)

    def gci0(self, g, rows, neg=False):
        if not self.context_axioms:
            return super().gci0(g, rows, neg)
        cm = g.class_model
        a = cm.in_context(rows[:, 0], cm.subclass_id)
        b = cm.in_context(rows[:, 1], cm.subclass_id)
        if self._neg(neg, "gci0"):
            return g.measure.apart(a, b, self._nm("gci0"))
        return g.measure.inside(a, b, self.m)

    def gci2(self, g, rows, neg=False):
        if not self.context_axioms:
            return super().gci2(g, rows, neg)
        cm = g.class_model
        a = cm.in_context(rows[:, 2], rows[:, 1])
        b = cm.in_context(rows[:, 0], rows[:, 1])
        if neg:
            return g.measure.apart(a, b, self._nm("gci2"))
        return g.measure.inside(a, b, self.m)

    def object_property_assertion(self, g, rows, neg=False):
        if self.assertion != "distance":
            return super().object_property_assertion(g, rows, neg)
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        image = g.relation_model.image(rows[:, 1], a)
        return th.linalg.norm(image.centre - b.centre, dim=1, keepdim=True)
