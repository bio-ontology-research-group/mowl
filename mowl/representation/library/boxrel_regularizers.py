"""Penalties of the boxes group, as :class:`mowl.training.Regularizer` objects: they keep
boxes in a cube (BoxEL, BoxSEL) or small (ELK-Em)."""

import torch as th

from mowl.training.regularization import Regularizer

__all__ = ["LocationPenalty", "VolumePenalty", "SizeCapPenalty"]


class LocationPenalty(Regularizer):
    """Keeps the boxes of the class model inside the cube :math:`[0, \\beta]^n`:
    :math:`\\operatorname{mean}\\max(0, M - \\beta + \\epsilon) +
    \\operatorname{mean}\\max(0, -m - \\epsilon)` over all entries of the upper and lower
    corners (:math:`L_{loc}` of BoxSEL, Eq. 9; with :math:`\\beta = 1` the regulariser of
    BoxEL). Needs a class model that stores a lower corner and log-widths
    (:class:`~mowl.representation.library.boxrel_boxes.MinDeltaBoxes`).

    :param beta: Side of the cube.
    """

    def __init__(self, beta=1.0, epsilon=1e-8, weight=1.0):
        super().__init__("class_embed", None, weight)
        self.beta = beta
        self.epsilon = epsilon

    def penalty(self, model):
        classes = getattr(model, "module", model).class_model
        low = classes.mins.weight
        high = low + th.exp(classes.deltas.weight)
        upper = th.relu(high - self.beta + self.epsilon).mean()
        lower = th.relu(-low - self.epsilon).mean()
        return self.weight * (upper + lower)


class VolumePenalty(Regularizer):
    """:math:`L_{vol}` of BoxSEL (Eq. 10, as printed): every box should have at least the
    volume of the cube, :math:`\\operatorname{mean}_C\\max(0, \\beta^n - \\mathrm{Vol}(C) -
    \\epsilon)`."""

    def __init__(self, beta=1.0, epsilon=1e-8, weight=1.0):
        super().__init__("class_embed", None, weight)
        self.beta = beta
        self.epsilon = epsilon

    def penalty(self, model):
        deltas = getattr(model, "module", model).class_model.deltas.weight
        volume = th.exp(deltas).prod(-1)
        return self.weight * th.relu(self.beta ** deltas.shape[1] - volume - self.epsilon).mean()


class SizeCapPenalty(Regularizer):
    """The box-size penalty of the authors' ELK-Em code:
    :math:`\\operatorname{mean}\\max(0, o - \\text{cap})^2` over the half-widths of the
    trained class boxes, role ranges and role error boxes. Needs
    :class:`~mowl.representation.library.boxrel_boxes.FixableBoxes` classes and
    :class:`~mowl.representation.library.boxrel_relations.AffineBoxes` relations."""

    def __init__(self, class_cap=0.25, range_cap=0.25, error_cap=0.05, weight=1.0):
        super().__init__("class_embed", None, weight)
        self.class_cap, self.range_cap, self.error_cap = class_cap, range_cap, error_cap

    def penalty(self, model):
        module = getattr(model, "module", model)
        classes, relations = module.class_model, module.relation_model
        free = ~(classes.bot_mask | classes.top_mask)
        class_o = classes.offsets.weight[free].abs()
        terms = [th.relu(relations.range_offsets.weight.abs() - self.range_cap).pow(2).mean(),
                 th.relu(relations.error_offsets.weight.abs() - self.error_cap).pow(2).mean()]
        if class_o.numel():
            terms.append(th.relu(class_o - self.class_cap).pow(2).mean())
        return self.weight * sum(terms)
