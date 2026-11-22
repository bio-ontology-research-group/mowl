r"""Gumbel boxes: expected volume, intersection, conditional probability, point distance.

Huang et al. (2023), Equations 4, 8 and 9. Used by the measures
:class:`~mowl.representation.library.shapes_measures.GumbelVolumeRatio` and
:class:`~mowl.representation.library.shapes_measures.PointToRegion` (the baseline
:mod:`mowl.baselines.nn.el.concept2box.losses` has its own copy).
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


def inside_weight(log_volume, alpha):
    r""":math:`\alpha(c) = \alpha \cdot \mathrm{sigmoid}(1 / \mathrm{Vol}(c))` (Equation 9),
    from the log of the expected volume."""
    return alpha * th.sigmoid(th.exp((-log_volume).clamp(max=60.0)))
