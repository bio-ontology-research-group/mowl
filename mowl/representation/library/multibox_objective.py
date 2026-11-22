"""The regularisers of the box-with-bump and multi-box family (group ``multibox``).

The readings of Box2EL, MultiBoxEL and MultiELBE are options of the core
:class:`~mowl.representation.Objective` (``squared``, ``gci2_negative="delta"``,
``gci3_bot_negative="filler"``) and of the relation model (:class:`HeadTailRelations`:
``subsumed`` with its two conditions, and the role axioms), so that there is no objective or
geometry class of this group.
"""

import torch as th

from mowl.nn.el.softagg import diversity_penalty
from mowl.training.regularization import NormPenalty, Regularizer

__all__ = ["bump_penalty", "ComponentSpread", "DiversityPenalty"]


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
