"""Differentiable fuzzy :math:`\\mathcal{ALC}` (DF-ALC, Wu et al. 2022), put together from parts.

The interpretation itself is the set of parameters: every concept name is a vector of
truth values over a finite domain :math:`\\Delta`, every role name an
:math:`|\\Delta| \\times |\\Delta|` matrix of truth values. Truth values are
``sigmoid(logit)`` so that they stay in :math:`[0, 1]`. Complex concepts use the Gödel
semantics of fuzzy :math:`\\mathcal{ALC}` (Eqs. 1-5 of the paper).

The module is the geometry of the method :func:`~mowl.representation.library.logic_fuzzy.dfalc`
(class model :class:`~mowl.representation.library.logic_fuzzy.TruthTableClasses`, relation
model :class:`~mowl.representation.library.logic_fuzzy.TruthMatrix`, measure
:class:`~mowl.representation.library.logic_fuzzy.MeanViolation`, objective
:class:`~mowl.representation.library.logic_fuzzy.FuzzyObjective`); the hand-written loss module
it replaces is kept for comparison as :mod:`mowl.baselines.nn.alc.dfalc`.
"""
import torch as th

from mowl.representation.library.logic_alc import ExpressionGeometry
from mowl.representation.library.logic_fuzzy import (dfalc_parts, has_quantifier,
                                                     rule_based_loss, rule_nf4, rule_nf5,
                                                     rule_nf6, rule_nf7)

__all__ = ["DFALCModule", "has_quantifier", "rule_based_loss", "rule_nf4", "rule_nf5",
           "rule_nf6", "rule_nf7", "hierarchical_loss"]


def hierarchical_loss(c, d):
    """Eq. 8 of the paper for one axiom :math:`C \\sqsubseteq D`: the mean over the domain
    of :math:`\\max(0, C^I(a) - D^I(a))`, one value per row (the ``inside`` of
    :class:`~mowl.representation.library.logic_fuzzy.MeanViolation`)."""
    return th.relu(c - d).mean(dim=-1)


class DFALCModule(ExpressionGeometry):
    """DF-ALC module: truth-value tables of concepts and roles over a finite domain.

    :param nclasses: Number of classes (including ``owl:Thing``, ``owl:Nothing``).
    :param nrelations: Number of object properties.
    :param ndomain: Size :math:`|\\Delta|` of the domain.
    :param top_index: Class index of ``owl:Thing`` (constant ``1``).
    :param bottom_index: Class index of ``owl:Nothing`` (constant ``0``).
    :param loss_type: ``"hierarchical"`` (Eq. 8, default) or ``"rule"``: Eq. 9 for
        inclusions without quantifiers, Eqs. 10-13 for the shapes of normal forms 4-7, Eq. 8
        for the others.
    :param threshold: The threshold :math:`\\alpha = \\alpha'` of the rule-based losses of
        normal forms 4-7 (paper: in :math:`[0.5, 1]`).
    :param init_std: Standard deviation of the logit initialisation.
    """

    def __init__(self, nclasses, nrelations, ndomain, top_index=None, bottom_index=None,
                 loss_type="hierarchical", init_std=1.0, threshold=0.8):
        if ndomain < 1:
            raise ValueError("The domain of a DF-ALC interpretation must not be empty.")
        if loss_type not in ("hierarchical", "rule"):
            raise ValueError("loss_type must be 'hierarchical' or 'rule'.")
        super().__init__(**dfalc_parts(nclasses, nrelations, ndomain, ndomain, ndomain,
                                       loss_type, threshold, "godel", top_index,
                                       bottom_index, init_std))
        self.ndomain = ndomain
        self.top_index = top_index
        self.bottom_index = bottom_index
        self.loss_type = loss_type
        self.threshold = threshold

    # -- groundings ----------------------------------------------------------------------
    def concept_truth(self, idx):
        """Truth values of class indices ``idx`` (shape ``idx.shape + (n,)``)."""
        return self.class_model(idx).values

    def role_truth(self, idx):
        """Truth values of role indices ``idx``, shape ``idx.shape + (n, n)``."""
        return self.relation_model.matrix(idx)
