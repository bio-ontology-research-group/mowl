"""Training regimes that belong to the shapes group: how the prism method combines its scores
and how the octagon method decays its learning rate.

Both are ordinary :class:`~mowl.training.TrainingRegime` objects: they can be attached to any
model that consults the hooks they change, and the methods that were published with them
name them in their builder (``default_regimes`` of the geometry,
:mod:`~mowl.representation.library.shapes_methods`).
"""

from mowl.training.losses import MSETargetLoss
from mowl.training.schedules import _Schedule

__all__ = ["SquaredMeanLoss", "ExponentialDecay"]


class SquaredMeanLoss(MSETargetLoss):
    r"""The loss of the prism method: the squared error of the *mean* score of each axiom form
    to its target, :math:`(\bar s^+ - t^+)^2 + (\bar s^- - t^-)^2` (positives target 0,
    negatives target 1, as ELBE does), where the positive term is not squared for
    ``square_positive=False``: the structural reading of the prism method has squared hinges
    in its scores already and only averages them, :math:`\bar s^+ + (\bar s^- - t^-)^2`.

    :param pos_target: Target of the positive scores.
    :param neg_target: Target of the negative scores.
    :param square_positive: Whether the mean positive score is squared.
    """

    def __init__(self, pos_target=0.0, neg_target=1.0, square_positive=True,
                 normal_forms=None):
        super().__init__(pos_target=pos_target, neg_target=neg_target, reduce="mean_first",
                         normal_forms=normal_forms)
        self.square_positive = bool(square_positive)

    def combine(self, gci_name, pos, neg):
        loss = self._term(pos, self.pos_target) if self.square_positive else pos.mean()
        if neg is not None:
            loss = loss + self._term(neg, self.neg_target)
        return loss


class ExponentialDecay(_Schedule):
    r"""The learning rate of epoch :math:`e` is the initial one times ``gamma ** e``.

    Hinge losses that must reach exactly zero (equalities such as
    :math:`A \equiv X_1 \sqcap X_2`) need it: with a constant learning rate Adam oscillates
    around the boundary (the octagon method's ``lr_decay``).

    :param gamma: The factor per epoch.
    """

    key = "exponential-lr-decay"

    def __init__(self, gamma=0.99, base_optimizer=None):
        super().__init__(base_optimizer)
        self.gamma = float(gamma)

    def setup(self, opt):
        self.base_lrs = [g["lr"] for g in opt.param_groups]

    def update(self, epoch, epochs):
        for g, lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = lr * self.gamma ** epoch
