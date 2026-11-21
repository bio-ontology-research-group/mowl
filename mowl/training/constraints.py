"""Regimes that treat loss terms as constraints.

A model trained on several loss terms (facts, axioms, region inclusions) is a model of
the ontology only when *every* term reaches zero. With fixed weights a large term can
hide a small one that never gets there. A Lagrangian treatment gives every term a weight
(a dual variable) that grows while the term is violated.
"""

from collections.abc import Mapping

from mowl.training.regime import TrainingRegime


def dual_ascent_step(weights, losses, rho=1.0, max_weight=1e3, target=0.0):
    """One dual-ascent step, in place:
    ``w[t] <- min(max_weight, w[t] + rho * max(0, losses[t] - target))`` for every term
    ``t`` of ``weights`` that ``losses`` has a value for.

    :param weights: Mutable mapping from term name to weight.
    :param losses: Mapping from term name to the term's mean loss (float or 0-d tensor).
    :return: ``weights``.
    """
    if not isinstance(losses, Mapping):
        raise TypeError(f"losses must be a mapping, got {type(losses).__name__}")
    for term in weights:
        if term not in losses:
            continue
        value = losses[term]
        value = float(value.detach().item()) if hasattr(value, "detach") else float(value)
        excess = max(0.0, value - target)
        weights[term] = float(min(max_weight, weights[term] + rho * excess))
    return weights


class LagrangianDualWeights(TrainingRegime):
    """Weight the loss terms of a model by dual variables updated by dual ascent after
    every epoch: the weight of a term grows by ``rho`` times the amount by which the
    term exceeds ``target``, up to ``max_weight``, and never shrinks.

    The model must report its loss per term (the ``losses`` of
    :meth:`~mowl.training.TrainingRegime.on_epoch_end`) and expose the weights it
    multiplies the terms with as a mutable mapping ``model.dual_weights`` with the same
    term names. GroundedEL does (terms ``"fact"``, ``"axioms"``, ``"region"``).

    :param rho: Step size of the dual ascent (>= 0).
    :param max_weight: Upper bound of every weight.
    :param target: Loss value below which a term counts as satisfied.
    :param terms: Terms to update; default: all terms of ``model.dual_weights``.
    """

    key = "lagrangian-dual-weights"

    def __init__(self, rho=1.0, max_weight=1e3, target=0.0, terms=None):
        super().__init__()
        if rho < 0 or max_weight < 1.0 or target < 0:
            raise ValueError("rho and target must be >= 0 and max_weight >= 1.")
        self.rho, self.max_weight, self.target = float(rho), float(max_weight), float(target)
        self.terms = None if terms is None else tuple(terms)
        self.history = []

    def attach(self, model):
        if not isinstance(getattr(model, "dual_weights", None), Mapping):
            raise TypeError(
                f"LagrangianDualWeights needs a model with a 'dual_weights' mapping; "
                f"{type(model).__name__} has none.")
        super().attach(model)
        # The model multiplies its terms with the weights only in this mode.
        if hasattr(model, "constraint_mode"):
            model.constraint_mode = "lagrangian"

    def on_epoch_end(self, epoch, losses):
        weights = self.model.dual_weights
        if self.terms is not None:
            selected = {t: weights[t] for t in self.terms}
            dual_ascent_step(selected, losses, self.rho, self.max_weight, self.target)
            weights.update(selected)
        else:
            dual_ascent_step(weights, losses, self.rho, self.max_weight, self.target)
        self.history.append(dict(weights))
