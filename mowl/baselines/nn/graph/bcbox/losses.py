"""Energies and losses of binary code box embeddings [zhang2022bcbox]_.

All functions follow the notation of Zhang, Boratko, Musco and McCallum, "Modeling
Transitivity and Cyclicity in Directed Graphs via Binary Code Box Embeddings" (NeurIPS
2022), Sections 2.2 and 4.

A box is given per dimension by its endpoints :math:`(x^-, x^+)`. The energy of a directed
edge :math:`(u, v)` is :math:`E(u, v) = -\\log P((u, v) \\in E)` and is low when the box of
``v`` lies inside the box of ``u`` (the score is normalised by the volume of ``v``, Eq. 3).
Edges therefore point from the larger (more general) box to the smaller one.
"""

import math

import torch as th

from mowl.baselines.nn.numerics import log_scaled_softplus
from mowl.baselines.nn.numerics import neg_log1mexp as _neg_log1mexp


def lse(a, b, t):
    """Two-argument LogSumExp with temperature, Eq. (4) of the paper.

    :math:`\\mathrm{LSE}_t(a, b) = t \\log(e^{a/t} + e^{b/t})` for :math:`t > 0` and
    :math:`\\max(a, b)` for :math:`t = 0`.

    :param a: First argument.
    :type a: :class:`torch.Tensor`
    :param b: Second argument (broadcastable to ``a``).
    :type b: :class:`torch.Tensor` or float
    :param t: Temperature :math:`t \\ge 0`.
    :type t: float
    :rtype: :class:`torch.Tensor`
    """
    if not th.is_tensor(b):
        b = th.full_like(a, float(b))
    a, b = th.broadcast_tensors(a, b)
    if t == 0:
        return th.maximum(a, b)
    return t * th.logsumexp(th.stack((a / t, b / t)), dim=0)


def log_lse0(s, t):
    """:math:`\\log \\mathrm{LSE}_t(s, 0)`, computed stably.

    For :math:`t > 0`, :math:`\\mathrm{LSE}_t(s, 0) = t\\,\\mathrm{softplus}(s/t)`. For very
    negative ``s`` the softplus underflows; there :math:`\\log \\mathrm{softplus}(z)
    \\approx z`, which is used instead. For :math:`t = 0` the value is
    :math:`\\log \\max(s, 0)` and may be :math:`-\\infty`.

    :param s: Side lengths.
    :type s: :class:`torch.Tensor`
    :param t: Volume temperature :math:`t \\ge 0`.
    :type t: float
    :rtype: :class:`torch.Tensor`
    """
    if t == 0:
        return th.log(th.clamp(s, min=0.0))
    return log_scaled_softplus(s / t, t)


def gumbel_box_log_score(x_min, x_max, y_min, y_max, intersection_temp, volume_temp):
    """Per-dimension :math:`\\log F_{\\text{G-Box}}` of Eq. (3).

    .. math::
        F_{\\text{G-Box}}((x^-, x^+), (y^-, y^+); (\\tau, \\nu)) =
        \\frac{\\mathrm{LSE}_\\nu(-\\mathrm{LSE}_\\tau(-x^+, -y^+)
               - \\mathrm{LSE}_\\tau(x^-, y^-), 0)}{\\mathrm{LSE}_\\nu(y^+ - y^-, 0)}

    With :math:`\\tau = \\nu = 0` this is the hard box score :math:`F_{\\text{Box}}` of
    Vilnis et al. (2018). The score is at most 1, so the returned value is at most 0.

    :param x_min: Lower endpoints of the first boxes (the edge source ``u``), shape ``(*, d)``.
    :param x_max: Upper endpoints of the first boxes, shape ``(*, d)``.
    :param y_min: Lower endpoints of the second boxes (the edge target ``v``), shape ``(*, d)``.
    :param y_max: Upper endpoints of the second boxes, shape ``(*, d)``.
    :param intersection_temp: :math:`\\tau \\ge 0`.
    :type intersection_temp: float
    :param volume_temp: :math:`\\nu \\ge 0`.
    :type volume_temp: float
    :returns: :math:`\\log F` per dimension, shape ``(*, d)``.
    :rtype: :class:`torch.Tensor`
    """
    meet_max = -lse(-x_max, -y_max, intersection_temp)
    meet_min = lse(x_min, y_min, intersection_temp)
    return log_lse0(meet_max - meet_min, volume_temp) - log_lse0(y_max - y_min, volume_temp)


def _code_weighted(neg_log_score, codes):
    """``codes * neg_log_score`` with :math:`F^0 = 1`: a dimension with code exactly 0 is
    ignored even where the hard score is 0 (``-log F = inf``), instead of giving ``nan``."""
    weighted = codes * neg_log_score
    return th.where(codes == 0, th.zeros_like(weighted), weighted)


def gbox_energy(neg_log_score):
    """G-Box energy, Eq. (2): :math:`-\\log \\prod_i F_i = \\sum_i -\\log F_i`.

    :param neg_log_score: :math:`-\\log F_i`, shape ``(*, d)``.
    :type neg_log_score: :class:`torch.Tensor`
    :rtype: :class:`torch.Tensor`
    """
    return neg_log_score.sum(dim=-1)


def gbc_energy(neg_log_score, codes):
    """GBC-Box energy, Eq. (8).

    .. math::
        E(u, v) = \\min_{j} -\\log \\prod_i F_i^{B_{i,j}}
                = \\min_j \\sum_i B_{i,j} (-\\log F_i)

    :param neg_log_score: :math:`-\\log F_i`, shape ``(*, d)``.
    :type neg_log_score: :class:`torch.Tensor`
    :param codes: Global binary code matrix :math:`B \\in [0, 1]^{d \\times k}`.
    :type codes: :class:`torch.Tensor`
    :rtype: :class:`torch.Tensor`
    """
    per_code = _code_weighted(neg_log_score.unsqueeze(-1), codes).sum(dim=-2)
    return per_code.min(dim=-1).values


def vbc_energy(neg_log_score, codes_u, codes_v):
    """VBC-Box energy, Eqs. (2) and (9).

    .. math::
        E(u, v) = -\\log \\prod_i F_i^{x^\\diamond_i y^\\diamond_i}
                = \\sum_i x^\\diamond_i y^\\diamond_i (-\\log F_i)

    :param neg_log_score: :math:`-\\log F_i`, shape ``(*, d)``.
    :type neg_log_score: :class:`torch.Tensor`
    :param codes_u: Per-vertex codes of ``u``, shape ``(*, d)``, values in ``[0, 1]``.
    :type codes_u: :class:`torch.Tensor`
    :param codes_v: Per-vertex codes of ``v``, shape ``(*, d)``, values in ``[0, 1]``.
    :type codes_v: :class:`torch.Tensor`
    :rtype: :class:`torch.Tensor`
    """
    return _code_weighted(neg_log_score, codes_u * codes_v).sum(dim=-1)


def neg_log1mexp(energy):
    """:math:`-\\log(1 - e^{-E})` for :math:`E \\ge 0`, computed stably (see
    :func:`mowl.nn.numerics.neg_log1mexp`).

    This is the negative log-probability that an edge with energy ``E`` is absent.

    :param energy: Energies, :math:`E \\ge 0`.
    :type energy: :class:`torch.Tensor`
    :rtype: :class:`torch.Tensor`
    """
    return _neg_log1mexp(energy)


def bce_negative_sampling_loss(pos_energy, neg_energy, neg_mask=None,
                               self_adversarial=True, adversarial_temperature=1.0):
    """Binary cross-entropy objective of Eq. (10), one value per positive edge.

    For a positive edge with energy :math:`E^+` and negatives :math:`E^-_1, \\dots,
    E^-_n`, the loss is :math:`E^+ + \\sum_j w_j \\cdot (-\\log(1 - e^{-E^-_j}))`.

    With ``self_adversarial=False`` all :math:`w_j = 1` (the plain sum of Eq. 10). With
    ``self_adversarial=True`` the weights are the self-adversarial weights of Sun et al.
    (RotatE, 2019), which the paper says it uses (Section 4.4): a softmax over the
    negatives of ``adversarial_temperature`` times the logit of the predicted edge
    probability, treated as constants (no gradient).

    :param pos_energy: Energies of the positive edges, shape ``(b,)``.
    :type pos_energy: :class:`torch.Tensor`
    :param neg_energy: Energies of the negatives, shape ``(b, n)``.
    :type neg_energy: :class:`torch.Tensor`
    :param neg_mask: Boolean mask of valid negatives, shape ``(b, n)``; invalid ones
        (for example a corruption that hit a true edge) get weight 0.
    :type neg_mask: :class:`torch.Tensor`, optional
    :param self_adversarial: Use self-adversarial negative weights.
    :type self_adversarial: bool
    :param adversarial_temperature: Inverse temperature :math:`\\alpha` of the weights.
    :type adversarial_temperature: float
    :rtype: :class:`torch.Tensor`
    """
    if neg_mask is None:
        neg_mask = th.ones_like(neg_energy, dtype=th.bool)
    neg_loss = neg_log1mexp(neg_energy)
    if self_adversarial:
        with th.no_grad():
            # logit of P(edge) = log p - log(1 - p), with log p = -E
            logit = -neg_energy + neg_loss
            logit = logit.masked_fill(~neg_mask, float("-inf"))
            weights = th.softmax(adversarial_temperature * logit, dim=-1)
            weights = th.nan_to_num(weights, nan=0.0)
    else:
        weights = neg_mask.to(neg_energy.dtype)
    neg_term = (weights * neg_loss.masked_fill(~neg_mask, 0.0)).sum(dim=-1)
    return pos_energy + neg_term


def code_lasso(codes):
    """Sparsity regulariser of Section 4.5: :math:`\\lVert \\mathbf{1} - B \\rVert_{\\ell_1}`.

    It penalises dropping dimensions (codes below 1). The model multiplies it by
    :math:`w_r`.

    :param codes: Trainable binary codes, values in ``[0, 1]``.
    :type codes: :class:`torch.Tensor`
    :rtype: :class:`torch.Tensor`
    """
    return (1.0 - codes).abs().sum()
