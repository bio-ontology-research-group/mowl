"""Reconstruction of a graph from the energies of its candidate edges."""

import numpy as np
import torch as th

__all__ = ["best_f1"]


def best_f1(energies, adjacency):
    """Best F1 over all global thresholds :math:`T` of the rule :math:`E < T`.

    This is the reconstruction measure of the paper (Section 5.1: "optimal F1 score").
    Infinite energies (for example the diagonal) are never predicted as edges.

    :param energies: Energy of every candidate pair, any shape.
    :type energies: :class:`torch.Tensor`
    :param adjacency: Boolean tensor of the same shape marking the true edges.
    :type adjacency: :class:`torch.Tensor`
    :returns: ``(f1, threshold)``.
    :rtype: tuple
    """
    e = energies.flatten().double()
    y = adjacency.flatten().bool()
    finite = th.isfinite(e)
    e, y_f = e[finite], y[finite]
    positives = int(y.sum())
    if positives == 0 or len(e) == 0:
        return 0.0, float("nan")
    order = th.argsort(e)
    e, y_f = e[order], y_f[order]
    tp = th.cumsum(y_f.double(), 0)
    k = th.arange(1, len(e) + 1, dtype=th.double)
    f1 = 2 * tp / (k + positives)
    # only cut between distinct energy values
    distinct = th.ones_like(y_f)
    distinct[:-1] = e[1:] > e[:-1]
    f1 = th.where(distinct, f1, th.zeros_like(f1))
    best = int(th.argmax(f1))
    upper = e[best + 1].item() if best + 1 < len(e) else float("inf")
    threshold = (e[best].item() + upper) / 2 if np.isfinite(upper) else e[best].item() + 1.0
    return float(f1[best]), threshold
