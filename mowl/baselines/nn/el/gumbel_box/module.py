r"""Gumbel box embeddings (Dasgupta et al., NeurIPS 2020) as a module over the class taxonomy."""
import math

import torch as th

from mowl.baselines.nn.el.box_lattice.module import UnitCubeBoxModule
from mowl.baselines.nn.numerics import log_scaled_softplus, neg_log1mexp

EULER_GAMMA = 0.5772156649015329


def softplus_volume(x, beta):
    r"""The softplus approximation of the expected side length
    :math:`m(x) = 2\beta K_0(2 e^{-x/(2\beta)})` (Section 4.2):
    :math:`\beta \log(1 + \exp(x / \beta - 2\gamma))`, with :math:`x = \mu_\vee - \mu_\wedge`."""
    return beta * th.nn.functional.softplus(x / beta - 2.0 * EULER_GAMMA)


def log_softplus_volume(x, beta, guard="stable"):
    """``log`` of :func:`softplus_volume`.

    :param guard: ``"stable"`` (default): :func:`mowl.nn.numerics.log_scaled_softplus`, the
        guard that the binary code boxes use, which keeps the exact value up to a relative
        error below ``1.5e-7`` and a gradient for any ``x``. ``"epsilon"``: add ``1e-30``
        inside the logarithm (earlier versions), which differs from the exact value for
        ``x / beta - 2 * gamma < -15`` (by at most ``1.5e-7``) and returns the constant
        ``log(beta) - 69`` with zero gradient below about ``-69``.
    """
    z = x / beta - 2.0 * EULER_GAMMA
    if guard == "epsilon":
        return math.log(beta) + th.log(th.nn.functional.softplus(z) + 1e-30)
    if guard != "stable":
        raise ValueError("guard must be 'stable' or 'epsilon'.")
    return log_scaled_softplus(z, beta)


class GumbelBoxModule(UnitCubeBoxModule):
    r"""Gumbel box embeddings of the class taxonomy.

    **From the paper.** The lower corner of a box is a MaxGumbel and the upper corner a
    MinGumbel random variable with location parameters :math:`\mu_\wedge, \mu_\vee` and
    common scale :math:`\beta` (Section 4). By Lemma 1 the intersection of boxes is again a
    Gumbel box with locations :math:`\mu_\wedge = \beta\,\mathrm{LSE}(\mu_{\wedge,k}/\beta)`
    and :math:`\mu_\vee = -\beta\,\mathrm{LSE}(-\mu_{\vee,k}/\beta)`. The expected side
    length of a box is :math:`m(\mu_\vee - \mu_\wedge)` (Proposition 1), which we compute
    with the softplus approximation of Section 4.2 (``m`` above), always positive, so
    disjoint boxes keep a gradient. The expected volume is the product of the side
    lengths, the conditional probability the ratio of two expected volumes (first-order
    Taylor approximation, Section 4.1) and training is cross entropy against the target
    conditional probability (Algorithm 1). The location parameters are kept in the unit
    cube (Section 4.1). (Algorithm 1 as printed has the signs of the two log-sum-exp
    terms swapped relative to Lemma 1; we follow Lemma 1.)

    **mOWL.** :math:`C \sqsubseteq D` is the target :math:`P(D \mid C) = 1`; the positive
    loss is :math:`-\log P(D \mid C)`, the negative loss :math:`-\log(1 - P(D \mid C))` for
    corrupted pairs (the paper samples negatives for its tree and graph experiments,
    Section 5; their number is the ``num_neg`` of the model). Unlike hard boxes, a Gumbel
    box has positive expected volume and :math:`P(D \mid C) < 1` also for identical boxes
    at finite :math:`\beta`, so the loss is only zero in the limit :math:`\beta \to 0`
    (where the model is the hard box lattice).

    :param beta: Gumbel scale :math:`\beta` (paper: 0.01 to 1e-4 depending on the task).
    :param min_delta: Minimum side :math:`\Delta_{\min}` of the location boxes.
    :param log_guard: Numerical guard of the logarithm of the side lengths, ``"stable"``
        (default) or ``"epsilon"`` (earlier versions); see :func:`log_softplus_volume`. It
        also selects the guard of the negative loss: ``"stable"`` is
        :func:`mowl.nn.numerics.neg_log1mexp` (the one of the binary code boxes, energy
        floor 1e-12), ``"epsilon"`` clamps :math:`\log P` at :math:`-10^{-9}`.
    """

    def __init__(self, nb_ont_classes, embed_dim=50, beta=0.01, min_delta=1e-6,
                 init_side=(0.3, 0.9), max_regularization=0.0, log_guard="stable"):
        if beta <= 0:
            raise ValueError("beta must be positive")
        if log_guard not in ("stable", "epsilon"):
            raise ValueError("log_guard must be 'stable' or 'epsilon'.")
        super().__init__(nb_ont_classes, embed_dim, min_delta, init_side, max_regularization)
        self.beta = float(beta)
        self.log_guard = log_guard

    def log_volume(self, mu_min, mu_max):
        """Log expected volume of the Gumbel box with the given location corners."""
        return log_softplus_volume(mu_max - mu_min, self.beta, self.log_guard).sum(dim=-1)

    def log_conditional(self, sub, sup, detach_delta=False):
        r""":math:`\log P(\text{sup} \mid \text{sub})` (always :math:`\le 0`)."""
        mn_a, mx_a = (t.double() for t in self.corners(sub, detach_delta))
        mn_b, mx_b = (t.double() for t in self.corners(sup, detach_delta))
        b = self.beta
        mn_i = b * th.logaddexp(mn_a / b, mn_b / b)
        mx_i = -b * th.logaddexp(-mx_a / b, -mx_b / b)
        return self.log_volume(mn_i, mx_i) - self.log_volume(mn_a, mx_a)

    def positive_loss(self, sub, sup):
        return (-self.log_conditional(sub, sup)).clamp(min=0.0).float()

    def negative_loss(self, sub, sup):
        logp = self.log_conditional(sub, sup)
        if self.log_guard == "stable":
            return neg_log1mexp(-logp).float()
        logp = logp.clamp(max=-1e-9)
        log1m = th.log(-th.expm1(logp))
        return (-log1m).float()
