"""How the costs of a pair become one training loss, for the graph-style hierarchy methods
(CatE and BCBox train on minibatches of edges with several corruptions per edge; the
generic loop of :class:`~mowl.base_models.EmbeddingELModel` does not fit them).

Both functions take the *costs* that a geometry returns (``geometry.inside`` for a stated
edge, the measure's ``negative`` or energy for a corruption), so the maths of the loss is
the same for every class model and measure that provides those costs.

==========================  ==========================================================
:func:`margin_loss`         CatE, Eq. 7: mean cost of the edges plus a hinge on the
                            cost of their corruptions
:func:`bce_edge_loss`       BCBox, Eq. 10: energy of the edge plus the weighted
                            :math:`-\\log(1 - e^{-E})` of the corruptions
==========================  ==========================================================
"""

import torch as th

from mowl.representation.library.hierarchy_measures import neg_log1mexp

__all__ = ["margin_loss", "bce_edge_loss"]


def margin_loss(pos_cost, neg_cost, margin, aggregation="per_negative", mask=None):
    """Mean cost of the stated edges plus a hinge of margin ``margin`` on the cost of the
    corrupted ones (CatE, Eq. 7; Vendrov et al. 2016, Eq. 4 with a mean).

    :param pos_cost: ``(b,)``, the cost :math:`s(A, B)` of the stated edges.
    :param neg_cost: ``(b, k)``, the cost of their ``k`` corruptions.
    :param aggregation: ``"per_negative"``: one hinge per corruption (the paper);
        ``"mean"``: the costs of the corruptions of an edge are averaged before the hinge
        (the reference code).
    :param mask: ``(b, k)`` boolean; ``False`` marks a corruption that is not a negative
        (it is a stated edge), left out of the means.
    :return: A scalar.
    """
    if neg_cost.dim() == 1:
        neg_cost = neg_cost.unsqueeze(1)
    if mask is None:
        mask = th.ones_like(neg_cost, dtype=th.bool)
    elif mask.dim() == 1:
        mask = mask.unsqueeze(1)
    w = mask.to(neg_cost.dtype)
    if aggregation == "per_negative":
        negative = (th.relu(margin - neg_cost) * w).sum() / w.sum().clamp(min=1)
    elif aggregation == "mean":
        count = w.sum(dim=1)
        mean_cost = (neg_cost * w).sum(dim=1) / count.clamp(min=1)
        valid = (count > 0).to(neg_cost.dtype)
        negative = (th.relu(margin - mean_cost) * valid).sum() / valid.sum().clamp(min=1)
    else:
        raise ValueError("aggregation must be 'per_negative' or 'mean'.")
    return pos_cost.mean() + negative


def bce_edge_loss(pos_energy, neg_energy, mask=None, self_adversarial=True, temperature=1.0,
                  floor=1e-12):
    """The binary cross entropy of binary code boxes (Zhang et al. 2022, Eq. 10), one value
    per stated edge: :math:`E^+ + \\sum_j w_j \\, (-\\log(1 - e^{-E^-_j}))`.

    :param pos_energy: ``(b,)``, the energy of the stated edges.
    :param neg_energy: ``(b, k)``, the energy of their corruptions.
    :param mask: ``(b, k)`` boolean; ``False`` gives a corruption weight 0.
    :param self_adversarial: Weights are a softmax over the corruptions of ``temperature``
        times the logit of the edge probability, treated as constants (Sun et al. 2019,
        RotatE); otherwise all weights are 1 (the plain sum).
    :param floor: Smallest energy of a corruption (keeps the loss finite).
    """
    if mask is None:
        mask = th.ones_like(neg_energy, dtype=th.bool)
    neg_loss = neg_log1mexp(neg_energy, floor)
    if self_adversarial:
        with th.no_grad():
            # logit of P(edge): log p - log(1 - p), with log p = -E
            logit = (-neg_energy + neg_loss).masked_fill(~mask, float("-inf"))
            weights = th.nan_to_num(th.softmax(temperature * logit, dim=-1), nan=0.0)
    else:
        weights = mask.to(neg_energy.dtype)
    return pos_energy + (weights * neg_loss.masked_fill(~mask, 0.0)).sum(dim=-1)
