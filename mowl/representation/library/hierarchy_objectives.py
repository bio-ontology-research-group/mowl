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


def _live_mean(values, live):
    """The mean of ``values`` over the entries where ``live`` is 1 (0 when there are none)."""
    return (values * live).sum() / live.sum().clamp(min=1)


def margin_loss(pos_cost, neg_cost, margin, aggregation="per_negative", mask=None):
    """A named objective for methods that score stated edges and their corruptions with the
    cost of a measure (:class:`~mowl.representation.library.hierarchy_measures.OrderViolation`
    and the other hierarchy measures): the mean cost of the stated edges plus the mean *hinge*
    :math:`\\max(0, \\gamma - \\text{cost})` of the corruptions, so that a stated edge is
    pulled to cost 0 and a corruption is pushed to cost at least ``margin`` (CatE, Eq. 7;
    Vendrov et al. 2016, Eq. 4 with means). A corruption that is itself a stated edge is
    masked out.

    The hinge of one corruption is the ``negative`` reading of the measure; this function
    adds the averaging over the batch and the two ways of averaging over the corruptions of
    one edge that the paper and its code use.

    :param pos_cost: ``(b,)``, the cost of the stated edges.
    :param neg_cost: ``(b, k)``, the cost of their ``k`` corruptions (``(b,)``: one each).
    :param margin: :math:`\\gamma \\ge 0`.
    :param aggregation: ``"per_negative"``: one hinge per corruption, averaged over all of
        them (the paper); ``"mean"``: the costs of the corruptions of an edge are averaged
        before the hinge, and the hinges over the edges that have corruptions (the code).
    :param mask: ``(b, k)`` boolean (or ``(b,)``); ``False`` marks a corruption that is
        not a negative.
    :return: A scalar.
    """
    if neg_cost.dim() == 1:
        neg_cost = neg_cost.unsqueeze(1)
    live = th.ones_like(neg_cost) if mask is None else (
        mask.reshape(neg_cost.shape)).to(neg_cost.dtype)
    if aggregation == "per_negative":
        corruptions = _live_mean(th.relu(margin - neg_cost), live)
    elif aggregation == "mean":
        per_edge = live.sum(dim=1)
        edge_cost = (neg_cost * live).sum(dim=1) / per_edge.clamp(min=1)
        corruptions = _live_mean(th.relu(margin - edge_cost), (per_edge > 0).to(live.dtype))
    else:
        raise ValueError("aggregation must be 'per_negative' or 'mean'.")
    return pos_cost.mean() + corruptions


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
