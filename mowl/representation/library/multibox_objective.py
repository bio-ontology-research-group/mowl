"""The objective, the geometry and the regularisers of the box-with-bump and multi-box family
(group ``multibox``).

:class:`MultiBoxObjective` is :class:`~mowl.representation.Objective` with the readings of
Box2EL, MultiBoxEL and MultiELBE added as options:

* ``squared``: the positive losses of the forms ``gci0``, ``gci0_bot``, ``gci1``,
  ``gci1_bot``, ``gci3`` and ``gci3_bot`` are squared (Box2EL).
* ``gci2_negative="delta"``: the negative loss is :math:`(\\delta - \\mu)^2` per argument
  (Box2EL and DELE), with a relation model that has ``subsumed_negative``.
* ``gci3_bot_negative``: ``"exists"`` (default; the region of :math:`\\exists r.C` is not
  empty) or ``"filler"`` (the size of :math:`C`, as the DELE paper writes it).
* the existentials use the relation model's own :meth:`subsumed`, when it has one, because
  with head and tail boxes :math:`C \\sqsubseteq \\exists r.D` has two conditions.
* negatives of ``apart`` use the margin in the same sense as the positives (a slack when
  the objective is a slack).
* the role forms ``role_inclusion`` and ``role_chain`` (:class:`MultiBoxGeometry`).
"""

import torch as th

from mowl.nn.el.softagg import diversity_penalty
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.objectives import Objective
from mowl.training.regularization import NormPenalty, Regularizer

__all__ = ["MultiBoxObjective", "MultiBoxGeometry", "bump_penalty", "ComponentSpread",
           "DiversityPenalty", "SQUARED_FORMS", "ROLE_FORMS"]

#: Forms squared by ``squared=True``.
SQUARED_FORMS = ("gci0", "gci1", "gci3", "gci0_bot", "gci1_bot", "gci3_bot")
#: The EL++ role axioms.
ROLE_FORMS = ("role_inclusion", "role_chain")


class MultiBoxObjective(Objective):
    """See the module. Other parameters as :class:`~mowl.representation.Objective`;
    ``gci2_negative`` also accepts ``"delta"``.

    :param squared: Square the positive loss of the forms in :data:`SQUARED_FORMS`.
    :param delta: The :math:`\\delta` of the ``"delta"`` negative loss.
    :param gci3_bot_negative: ``"exists"`` or ``"filler"``.
    """

    def __init__(self, margin=0.0, slack=False, epsilon=0.01, gci3="inclusion",
                 gci2_negative="apart", negatives="all", empty="size", gci2_offset=0.0,
                 squared=False, delta=2.0, gci3_bot_negative="exists"):
        reading = gci2_negative
        base_reading = "apart" if reading == "delta" else reading
        super().__init__(margin, slack, epsilon, gci3, base_reading, negatives, empty,
                         gci2_offset)
        self.gci2_negative = reading
        if reading not in ("apart", "overlap", "linear", "delta"):
            raise ValueError("gci2_negative must be 'apart', 'overlap', 'linear' or 'delta'.")
        if gci3_bot_negative not in ("exists", "filler"):
            raise ValueError("gci3_bot_negative must be 'exists' or 'filler'.")
        self.squared = bool(squared)
        self.delta = float(delta)
        self.gci3_bot_negative = gci3_bot_negative

    # -- squaring ---------------------------------------------------------------------------
    def cost(self, g, form, rows, neg=False):
        if form in ROLE_FORMS:
            return getattr(self, form)(g, rows)
        loss = super().cost(g, form, rows, neg)
        if self.squared and form in SQUARED_FORMS and not self._neg(neg):
            loss = loss.square()
        return loss

    # -- helpers ----------------------------------------------------------------------------
    @staticmethod
    def _overlap(measure, a, b, margin):
        """``meets`` with every pair of components counted (the negative reading)."""
        return getattr(measure, "meets_all", measure.meets)(a, b, margin)

    def _existential(self, g, sub, relation, filler, neg):
        rel = g.relation_model
        if not neg:
            if hasattr(rel, "subsumed"):
                cost = rel.subsumed(g.measure, sub, relation, filler, self.m)
            else:
                cost = g.measure.inside(sub, g.exists(relation, filler), self.m)
            return cost + self.gci2_offset
        if self.gci2_negative == "delta":
            return rel.subsumed_negative(g.measure, sub, relation, filler, self.margin,
                                         self.delta)
        target = g.exists(relation, filler)
        if self.gci2_negative == "overlap":
            return self._overlap(g.measure, sub, target, -self.margin)
        return super()._existential(g, sub, relation, filler, neg)

    # -- the forms --------------------------------------------------------------------------
    def gci0(self, g, rows, neg=False):
        if self._neg(neg):
            c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
            return g.measure.apart(c, d, self.m)
        return super().gci0(g, rows, neg)

    def gci1(self, g, rows, neg=False):
        if self._neg(neg):
            c, d, e = (g.classes(rows[:, i]) for i in range(3))
            return g.measure.intersection_apart(c, d, e, self.m)
        return super().gci1(g, rows, neg)

    def gci1_bot(self, g, rows, neg=False):
        if not self._neg(neg):
            c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
            return g.measure.apart(c, d, self.m)
        return super().gci1_bot(g, rows, neg)

    def gci3(self, g, rows, neg=False):
        if self._neg(neg) and self.gci2_negative == "delta":
            exists = g.exists(rows[:, 0], g.classes(rows[:, 1]))
            d = g.classes(rows[:, 2])
            return (self.delta - g.measure.meets(exists, d, self.margin)).square()
        return super().gci3(g, rows, neg)

    def gci3_bot(self, g, rows, neg=False):
        if self._neg(neg) and self.gci3_bot_negative == "filler":
            return g.measure.nonempty(g.classes(rows[:, 1]), self.epsilon)
        # exists r.C is empty iff the region of exists r.C is
        return self._bottom(g, g.exists(rows[:, 0], g.classes(rows[:, 1])), neg)

    def object_property_assertion(self, g, rows, neg=False):
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        return self._existential(g, a, rows[:, 1], b, neg)

    # -- role axioms ------------------------------------------------------------------------
    def role_inclusion(self, g, rows):
        """:math:`r \\sqsubseteq s`: head in head, tail in tail (the mean of the two)."""
        rel = g.relation_model
        r, s = rows[:, 0], rows[:, 1]
        head = g.measure.inside(rel.head(r), rel.head(s), self.m)
        tail = g.measure.inside(rel.tail(r), rel.tail(s), self.m)
        return (head + tail) / 2

    def role_chain(self, g, rows):
        """:math:`r_1 \\circ r_2 \\sqsubseteq s`: the head of :math:`r_1` in the head of
        :math:`s` and the tail of :math:`r_2` in the tail of :math:`s`."""
        rel = g.relation_model
        r1, r2, s = rows[:, 0], rows[:, 1], rows[:, 2]
        head = g.measure.inside(rel.head(r1), rel.head(s), self.m)
        tail = g.measure.inside(rel.tail(r2), rel.tail(s), self.m)
        return (head + tail) / 2


class MultiBoxGeometry(ComposedGeometry):
    """A :class:`~mowl.representation.ComposedGeometry` that also has the role axiom losses
    (``role_inclusion_loss`` and ``role_chain_loss``) when its relation model has head and
    tail boxes."""

    @property
    def role_axiom_capable(self):
        return hasattr(self.relation_model, "head") and hasattr(self.objective, "role_chain")

    def cost(self, form, rows, neg=False):
        return self.objective.cost(self, form, rows, neg)

    def get_loss_function(self, gci_name):
        if gci_name in ROLE_FORMS:
            return lambda rows, neg=False: self.objective.cost(self, gci_name, rows, neg)
        return super().get_loss_function(gci_name)

    def role_inclusion_loss(self, rows, neg=False):
        return self.objective.role_inclusion(self, rows)

    def role_chain_loss(self, rows, neg=False):
        return self.objective.role_chain(self, rows)


# ====================================================================================
# regularisers
# ====================================================================================

def bump_tables(model):
    """The bump tables of the classes (and individuals) of a geometry."""
    tables = [model.class_model.bumps.weight]
    individuals = model.individual_model
    if individuals is not None and getattr(individuals, "bumps", None) is not None:
        tables.append(individuals.bumps.weight)
    return tables


def bump_penalty(weight=0.05):
    """The regulariser of Box2EL: ``weight`` times the mean norm of the bump vectors (of the
    classes, and of the individuals if they have bumps)."""
    return NormPenalty(parameters=bump_tables, target=0.0, weight=weight)


class ComponentSpread(Regularizer):
    """Keeps the components of a class apart: :math:`\\operatorname{mean}_C
    \\operatorname{mean}_{i<j} \\max(0, \\tau - \\lVert c_{C,i} - c_{C,j} \\rVert)` of the
    centres of the boxes (the anti-collapse penalty of MultiBoxEL and MultiELBE).

    :param weight: Factor.
    :param separation: :math:`\\tau`.
    """

    kind = "penalty"

    def __init__(self, weight=1.0, separation=0.1):
        super().__init__(parameters=(), weight=weight)
        self.separation = separation

    def penalty(self, model):
        classes = model.class_model
        k = classes.components
        if k < 2 or self.weight == 0:
            return None
        centres = classes.centres.weight.view(len(classes), k, classes.embed_dim)
        i, j = th.triu_indices(k, k, offset=1, device=centres.device)
        dists = th.linalg.norm(centres[:, i, :] - centres[:, j, :], dim=-1)
        return self.weight * th.relu(self.separation - dists).mean(dim=1).mean(dim=0)


class DiversityPenalty(Regularizer):
    """The soft containment ratio between the components of every class
    (:func:`~mowl.nn.el.softagg.diversity_penalty`), which counters the collapse that the soft
    aggregation causes."""

    kind = "penalty"

    def __init__(self, weight=1.0):
        super().__init__(parameters=(), weight=weight)

    def penalty(self, model):
        classes = model.class_model
        k = classes.components
        if k < 2 or self.weight == 0 or len(classes) == 0:
            return None
        shape = (len(classes), k, classes.embed_dim)
        return self.weight * diversity_penalty(classes.centres.weight.view(shape),
                                               th.abs(classes.offsets.weight).view(shape))
