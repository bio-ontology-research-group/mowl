"""Probabilistic conditionals :math:`(D \\mid C)[l, u]` as a training term (BoxSEL, Zhu et al.
2024).

Statistical EL has a statement that OWL cannot express: a conditional probability bound. A
geometry that reads the form ``"conditional"`` (the volume ratio of two regions;
:class:`~mowl.nn.el.boxsel.module.ConditionalExpressions`) can be trained on such statements
with :class:`ConditionalStatements`, next to the axioms of the ontology; the regime is not
specific to boxes.
"""

from collections import namedtuple

import torch as th

from mowl.training.regime import TrainingRegime

_THING = "http://www.w3.org/2002/07/owl#Thing"

#: A statistical conditional :math:`(D \mid C)[l, u]`: ``conclusion`` is :math:`D`,
#: ``premise`` is :math:`C`. Concept expressions are a class IRI, ``("and", e1, e2, ...)``
#: or ``("some", role_iri, e)``.
Conditional = namedtuple("Conditional", ["conclusion", "premise", "lower", "upper"])


def as_conditional(item):
    """A :class:`Conditional` from one, a ``(D, C, p)`` or a ``(D, C, l, u)`` tuple."""
    if isinstance(item, Conditional):
        c = item
    elif len(item) == 3:                      # (D, C, p)
        c = Conditional(item[0], item[1], item[2], item[2])
    else:
        c = Conditional(*item)
    if not (0.0 <= c.lower <= c.upper <= 1.0):
        raise ValueError(f"A conditional needs 0 <= l <= u <= 1, got [{c.lower}, {c.upper}].")
    return c


def compile_conditional_expression(expr, class_index, property_index):
    """``(structure, leaf indices)`` of a concept expression; :math:`\\top` has no finite box."""
    from mowl.nn.el.boxsel import losses as S
    if expr == _THING:
        raise ValueError("owl:Thing has no finite box: conditionals over it are not "
                         "supported (the TBox must be safe, Definition 2).")
    return S.compile_expression(expr, class_index, property_index)


class ConditionalStatements(TrainingRegime):
    """The loss of probabilistic conditionals, added once per step: for every group of
    conditionals with the same structure of conclusion and premise the mean of
    ``module.conditional_loss`` (:math:`[l\\,\\mathrm{Vol}(C) - \\mathrm{Vol}(D \\cap C)]_+ +
    [\\mathrm{Vol}(D \\cap C) - u\\,\\mathrm{Vol}(C)]_+`), summed over the groups and multiplied
    by ``weight``.

    :param statements: Iterable of :class:`Conditional` or ``(D, C, l, u)`` / ``(D, C, p)``,
        or ``None``: the ``conditionals`` of the model (which may grow while the regime is
        attached; :meth:`add` adds to the regime's own list).
    :param weight: Weight of the term (``conditional_weight`` of BoxSEL).
    """

    key = "conditional-statements"

    def __init__(self, statements=None, weight=1.0):
        super().__init__()
        self.statements = None if statements is None else [as_conditional(c)
                                                            for c in statements]
        self.weight = weight
        self._groups = None
        self._count = -1

    def add(self, conditionals):
        """Add conditionals to the regime's own list."""
        if self.statements is None:
            self.statements = []
        self.statements.extend(as_conditional(c) for c in conditionals)

    def _current(self):
        if self.statements is not None:
            return self.statements
        return getattr(self.model, "conditionals", ())

    def groups(self):
        """The conditionals grouped by structure, as index tensors (compiled when the number
        of conditionals changes)."""
        current = self._current()
        if self._groups is None or self._count != len(current):
            model = self.model
            groups = {}
            for c in current:
                (sd, ld) = compile_conditional_expression(
                    c.conclusion, model.class_index_dict, model.object_property_index_dict)
                (sc, lc) = compile_conditional_expression(
                    c.premise, model.class_index_dict, model.object_property_index_dict)
                g = groups.setdefault((sd, sc), {"d": [], "c": [], "l": [], "u": []})
                g["d"].append(ld)
                g["c"].append(lc)
                g["l"].append(c.lower)
                g["u"].append(c.upper)
            device = model.device
            self._groups = {
                key: (th.tensor(g["d"], dtype=th.long, device=device),
                      th.tensor(g["c"], dtype=th.long, device=device),
                      th.tensor(g["l"], dtype=th.float, device=device),
                      th.tensor(g["u"], dtype=th.float, device=device))
                for key, g in groups.items()}
            self._count = len(current)
        return self._groups

    def additional_loss(self, epoch):
        if not self._current():
            return None
        module = self.model.module
        total = th.zeros((), device=self.model.device)
        for (sd, sc), (ld, lc, lo, up) in self.groups().items():
            total = total + module.conditional_loss(sd, sc, ld, lc, lo, up).mean()
        return self.weight * total
