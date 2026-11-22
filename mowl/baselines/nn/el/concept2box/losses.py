r"""Geometry and losses of Concept2Box [huang2023]_.

Concepts are *Gumbel boxes* (Dasgupta et al., 2020; Chen et al., 2021): a box is a centre
and a non-negative offset, with minimum corner :math:`\mu^m = c - o` and maximum corner
:math:`\mu^M = c + o`, whose corners are Gumbel-distributed with scale :math:`\beta`. The
expected volume and the expected intersection have closed forms (below) and are positive
and smooth for every pair of boxes, so that disjoint boxes still receive gradients.

All volumes are computed in log space. As :math:`\beta \to 0` they become the volume of
the hard box and the intersection of the hard boxes.

Entities are plain vectors. The distance from a vector to a box (Equation 8) is

.. math::

    f_d(e, c) = \lVert \max(e - \mu^M, 0) + \max(\mu^m - e, 0) \rVert_1
        + \alpha(c) \, \lVert \mathrm{cen}(c) - \min(\mu^M, \max(\mu^m, e)) \rVert_1,

with :math:`\alpha(c) = \alpha \cdot \mathrm{sigmoid}(1 / \mathrm{Vol}(c))` (Equation 9).
"""


import torch as th
import torch.nn.functional as F

EULER_GAMMA = 0.5772156649015329


def box_corners(center, offset):
    """Minimum and maximum corner of the box ``(center, offset)``."""
    return center - offset, center + offset


def gumbel_log_volume(mu_min, mu_max, beta):
    r"""Log of the expected volume of a Gumbel box,
    :math:`\sum_i \log\big(\beta \, \mathrm{softplus}((\mu^M_i - \mu^m_i)/\beta - 2\gamma)\big)`
    with :math:`\gamma` the Euler-Mascheroni constant."""
    side = beta * F.softplus((mu_max - mu_min) / beta - 2 * EULER_GAMMA)
    return th.log(side.clamp_min(1e-30)).sum(dim=-1)


def gumbel_intersection(min1, max1, min2, max2, beta):
    """Corners of the intersection of two Gumbel boxes: the soft maximum of the minimum
    corners and the soft minimum of the maximum corners."""
    lo = beta * th.logaddexp(min1 / beta, min2 / beta)
    hi = -beta * th.logaddexp(-max1 / beta, -max2 / beta)
    return lo, hi


def conditional_probability(min_h, max_h, min_t, max_t, beta):
    r"""Equation 4 of the paper: :math:`\mathrm{E}[\mathrm{Vol}(h \cap t)] /
    \mathrm{E}[\mathrm{Vol}(t)]`. Approximately 1 when box ``t`` is inside box ``h``."""
    lo, hi = gumbel_intersection(min_h, max_h, min_t, max_t, beta)
    return th.exp(gumbel_log_volume(lo, hi, beta) - gumbel_log_volume(min_t, max_t, beta))


def inside_weight(log_volume, alpha):
    r""":math:`\alpha(c) = \alpha \cdot \mathrm{sigmoid}(1 / \mathrm{Vol}(c))` (Equation 9),
    from the log of the expected volume."""
    return alpha * th.sigmoid(th.exp((-log_volume).clamp(max=60.0)))


def vector_to_box_distance(e, center, offset, alpha, beta):
    """Equation 8 (with the volume-dependent balancing of Equation 9): distance from the
    vector ``e`` to the box ``(center, offset)``. Zero outside part when ``e`` is in the
    box; the inside part is the distance to the centre, weighted by ``alpha(c)``."""
    mu_min, mu_max = box_corners(center, offset)
    out = (th.relu(e - mu_max) + th.relu(mu_min - e)).sum(dim=-1)
    inside_point = th.minimum(mu_max, th.maximum(mu_min, e))
    dist_in = (center - inside_point).abs().sum(dim=-1)
    log_volume = gumbel_log_volume(mu_min, mu_max, beta)
    return out + inside_weight(log_volume, alpha) * dist_in


def ontology_view_loss(phi, neg):
    """Equation 6, one term per row: :math:`|\\phi - 1|^2` for a true triple and
    :math:`|\\phi|^2` for a corrupted one."""
    return phi ** 2 if neg else (phi - 1) ** 2


def instance_view_loss(distance, neg, margin):
    """Instance view (Equation 7) with a translational model: the distance for a true
    triple, a hinge :math:`[\\gamma - d]_+` for a corrupted one (the paper's pairwise
    hinge, split into its positive and negative part)."""
    return F.relu(margin - distance) if neg else distance


def cross_view_loss(distance, neg, margin):
    r"""Cross view (Equation 10), split like the instance view: with
    :math:`s = \sigma(f_d) - 1/2 \in [0, 1/2)`, a true link costs :math:`s` and a corrupted
    one :math:`[\gamma - s]_+`."""
    s = th.sigmoid(distance) - 0.5
    return F.relu(margin - s) if neg else s
