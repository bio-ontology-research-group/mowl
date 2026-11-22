r"""Which axioms a fuzzy interpretation violates (satisfaction in fuzzy :math:`\mathcal{ALC}`).

A fuzzy interpretation over a finite domain satisfies :math:`C \sqsubseteq D` when
:math:`C^I(a) \le D^I(a)` for every domain element :math:`a` (Wu et al. 2022, Eq. 6). The
functions here measure how far a geometry whose classes are fuzzy sets (a
:class:`~mowl.representation.library.logic_alc.ExpressionGeometry` with a
:class:`~mowl.representation.library.logic_fuzzy.FuzzySet` region) is from that, per row, and
count the training axioms that are violated beyond a tolerance. They apply to any model with a
fuzzy class model, not only to one method.

=================================  ===================================================
:func:`row_violation`              the largest violation over the domain, per row
:func:`violated_axioms`            the axiom patterns of a model with violated rows
=================================  ===================================================
"""

import torch as th

from mowl.owlapi import OWLDisjointClassesAxiom, OWLEquivalentClassesAxiom, OWLSubClassOfAxiom

__all__ = ["row_violation", "violated_axioms"]


def row_violation(module, axiom, x):
    """The violation of the rows ``x`` of the axiom pattern ``axiom`` by the fuzzy
    interpretation of ``module``, one number per row (the worst element of the domain):

    * :math:`C \\sqsubseteq D`: :math:`\\max_a \\max(0, C(a) - D(a))`;
    * :math:`C \\equiv D`: :math:`\\max_a |C(a) - D(a)|`;
    * :math:`C \\sqcap D \\sqsubseteq \\bot` (disjointness): :math:`\\max_a \\min(C(a), D(a))`;
    * assertions: the cost the module gives them (the distance of the truth value to 1).
    """
    if isinstance(axiom, OWLSubClassOfAxiom):
        c, nxt = module.forward_fs(axiom.getSubClass(), x)
        d, _ = module.forward_fs(axiom.getSuperClass(), x, None, nxt)
        return th.relu(c - d).amax(-1)
    if isinstance(axiom, (OWLEquivalentClassesAxiom, OWLDisjointClassesAxiom)):
        expressions = axiom.getClassExpressionsAsList()
        c, nxt = module.forward_fs(expressions[0], x)
        d, _ = module.forward_fs(expressions[1], x, None, nxt)
        if isinstance(axiom, OWLEquivalentClassesAxiom):
            return (c - d).abs().amax(-1)
        return th.minimum(c, d).amax(-1)
    return module(axiom, x, None)


def violated_axioms(model, tolerance=1e-6):
    """The training axiom patterns of ``model`` (an ALC model with ``training_dataloaders``
    and ``axiom_category``) that have rows violated by more than ``tolerance``, as a list of
    ``(axiom, number of violating rows)``; assertions are violated when their truth value is
    below ``1 - tolerance``."""
    out = []
    with th.no_grad():
        for axiom, loader in model.training_dataloaders.items():
            if model.axiom_category(axiom) is None:
                continue
            bad = 0
            for batch in loader:
                x = batch[0].to(model.device)
                bad += int((row_violation(model.module, axiom, x) > tolerance).sum())
            if bad:
                out.append((axiom, bad))
    return out
