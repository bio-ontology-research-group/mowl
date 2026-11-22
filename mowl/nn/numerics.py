"""Numerical helpers shared by several modules."""
import math

import torch as th
import torch.nn.functional as F

#: Below this value of ``z``, ``log softplus(z)`` is replaced by ``z`` (its limit).
LOG_SOFTPLUS_LIMIT = -15.0


def log_softplus(z):
    """:math:`\\log \\mathrm{softplus}(z)`, finite and with a gradient for every ``z``.

    For very negative ``z`` the softplus underflows to 0 and its logarithm is
    :math:`-\\infty`; there :math:`\\log \\mathrm{softplus}(z) \\approx z` (relative error
    :math:`e^{z}/2`, below ``1.5e-7`` for ``z < -15``), which is used instead.
    """
    safe = th.clamp(z, min=LOG_SOFTPLUS_LIMIT)
    return th.where(z > LOG_SOFTPLUS_LIMIT, th.log(F.softplus(safe)), z)


def log_scaled_softplus(z, scale):
    """:math:`\\log(\\mathrm{scale} \\cdot \\mathrm{softplus}(z))`."""
    return math.log(scale) + log_softplus(z)


def neg_log1mexp(energy, floor=1e-12):
    """:math:`-\\log(1 - e^{-E})` for :math:`E \\ge 0`, computed stably.

    The energy is clamped at ``floor`` so that the loss of an edge that is almost certainly
    present stays finite (about ``-log(floor)``).
    """
    e = th.clamp(energy, min=floor)
    small = e < math.log(2.0)
    out_small = th.log(-th.expm1(-th.where(small, e, th.ones_like(e))))
    out_large = th.log1p(-th.exp(-th.where(small, th.ones_like(e), e)))
    return -th.where(small, out_small, out_large)
