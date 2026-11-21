r"""Formal disks (Suzuki et al. 2019).

A generalised formal disk is a pair :math:`(x, r)` of a centre in a space with a
quasi-metric :math:`d` and a real (possibly negative) radius. :math:`(x_i, r_i) \sqsupseteq
(x_j, r_j)` iff :math:`d(x_i, x_j) \le r_i - r_j`; the protrusion
:math:`l_{ij} = d(x_i, x_j) - r_i + r_j` (Eq. 9) is :math:`\le 0` exactly then.
"""

import torch as th


def protrusion(manifold, x_big, r_big, x_small, r_small):
    """:math:`l = d(x_{big}, x_{small}) - r_{big} + r_{small}`."""
    return manifold.distance(x_big, x_small) - r_big + r_small


def disk_loss(pos, neg):
    """:math:`\\sum h_+(E_{ij}) + \\sum h_+(\\mu - E_{ij})` (Eq. 10) with
    :math:`E = l` (Eq. 11), as means. ``pos`` are protrusions of positive pairs, ``neg``
    the already computed :math:`h_+(\\mu - l)` of the negatives."""
    loss = th.relu(pos).mean()
    if neg is not None:
        loss = loss + neg.mean()
    return loss
