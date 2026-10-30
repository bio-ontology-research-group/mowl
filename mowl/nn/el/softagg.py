"""Soft component aggregation and diversity regularisation helpers (Objective O1).

Provides temperature-controlled soft minimum and softmax weights over component
axes so that every component of a multi-box class receives gradient during
training, together with a pairwise soft containment volume penalty that
breaks collapsed or nested component configurations without pushing apart
components that are already disjoint.

Guarantees:
- **Exact k = 1 identity:** When the aggregated dimension has size ``1``,
  :func:`softmin` returns the single slice (to floating-point precision) and
  :func:`soft_weights` returns ``1.0`` for every positive temperature ``tau``.
- **Temperature limits:** For any tensor ``v``, :func:`softmin` converges to
  ``torch.amin(v, dim=dim)`` as :math:`\\tau \\to 0^+` (bounded within
  :math:`[\\min(v), \\min(v) + \\tau \\log K]`) and to ``v.mean(dim=dim)`` as
  :math:`\\tau \\to \\infty`.
- **Disjointness invariance of diversity penalty:** :func:`diversity_penalty`
  evaluates to ``0.0`` whenever all component pairs of a class are disjoint
  (or touch only on their boundary), and is ``0.0`` when ``n_components == 1``.

Non-guarantees:
- For finite :math:`\\tau > 0` and :math:`K > 1`, :func:`softmin` is a strict
  upper bound on ``torch.amin`` whenever components are not all identical
  (:math:`\\min(v) \\le \\text{softmin}_\\tau(v) \\le \\text{mean}(v)`).
"""
from __future__ import annotations

import math

import torch as th
import torch.nn.functional as F

__all__ = [
    "softmin",
    "soft_weights",
    "diversity_penalty",
]


def _validate_tau(tau: float) -> float:
    """Validate that temperature ``tau`` is a strictly positive finite float."""
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise ValueError(f"tau must be a positive number, got {tau!r}")
    tau_f = float(tau)
    if tau_f <= 0.0 or math.isnan(tau_f) or math.isinf(tau_f):
        raise ValueError(f"tau must be > 0 and finite, got {tau!r}")
    return tau_f


def softmin(
    v: th.Tensor,
    tau: float,
    dim: int = -1,
    keepdim: bool = False,
) -> th.Tensor:
    """Compute the normalised temperature soft-minimum along ``dim``.

    Uses the log-mean-exp formula

    .. math::

        \\text{softmin}_\\tau(v)
        = -\\tau \\left(\\log \\sum_{j=1}^{K} \\exp(-v_j / \\tau) - \\log K\\right)
        = -\\tau \\log \\left(\\frac{1}{K} \\sum_{j=1}^{K} \\exp(-v_j / \\tau)\\right),

    whose gradient with respect to :math:`v` is identically
    :math:`\\nabla_v (-\\tau \\,\\text{logsumexp}(-v / \\tau)) = \\text{softmax}(-v / \\tau)`
    while keeping :math:`\\text{softmin}_\\tau(v) \\in [\\min(v), \\text{mean}(v)]`
    and :math:`\\text{softmin}_\\tau(0) = 0`.

    Guarantees:
    - When ``v.shape[dim] == 1``, equals ``v.squeeze(dim)`` (or ``v`` when
      ``keepdim=True``) to floating-point precision for all ``tau > 0``.
    - Bounded between ``v.amin(dim=dim)`` and ``v.amin(dim=dim) + tau * log(K)``,
      and bounded above by ``v.mean(dim=dim)``.
    - Converges to ``v.amin(dim=dim)`` as ``tau -> 0+`` and to ``v.mean(dim=dim)``
      as ``tau -> inf``.
    - Every component along ``dim`` receives a strictly positive gradient weight
      ``softmax(-v / tau)``.

    Non-guarantees:
    - Raises :class:`ValueError` if ``tau <= 0``.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor of scores.
    tau : float
        Strictly positive temperature :math:`\\tau > 0`.
    dim : int, default -1
        Dimension along which to aggregate.
    keepdim : bool, default False
        Whether the output tensor retains ``dim`` with size ``1``.

    Returns
    -------
    torch.Tensor
        Soft-minimum tensor with ``dim`` reduced (or size ``1`` if ``keepdim=True``).
    """
    tau_f = _validate_tau(tau)
    k = v.shape[dim]
    if k == 1:
        return v if keepdim else v.squeeze(dim)
    lse = th.logsumexp(-v / tau_f, dim=dim, keepdim=keepdim)
    return -tau_f * (lse - math.log(k))


def soft_weights(
    v: th.Tensor,
    tau: float,
    dim: int = -1,
) -> th.Tensor:
    """Compute soft-argmin weights ``softmax(-v / tau, dim=dim)``.

    Guarantees:
    - Output has the same shape as ``v``, with strictly positive entries that
      sum to ``1`` along ``dim``.
    - When ``v.shape[dim] == 1``, returns a tensor of ones with the same shape
      as ``v``.
    - Concentrates weight on ``argmin(v, dim=dim)`` as ``tau -> 0+`` and
      approaches uniform ``1 / K`` weights as ``tau -> inf``.

    Non-guarantees:
    - Raises :class:`ValueError` if ``tau <= 0``.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor of scores.
    tau : float
        Strictly positive temperature :math:`\\tau > 0`.
    dim : int, default -1
        Dimension along which to normalise weights.

    Returns
    -------
    torch.Tensor
        Softmax weight tensor of the same shape as ``v``.
    """
    tau_f = _validate_tau(tau)
    return th.softmax(-v / tau_f, dim=dim)


def diversity_penalty(
    centers: th.Tensor,
    offsets: th.Tensor,
    eps: float = 1e-6,
) -> th.Tensor:
    """Compute the pairwise soft containment ratio across components of each class.

    For each class :math:`C` with :math:`k` box components
    :math:`C_i = [c_{C,i} - |o_{C,i}|,\\, c_{C,i} + |o_{C,i}|]`:

    .. math::

        \\mathcal{L}_{\\text{div}}
        = \\frac{1}{|\\mathcal{C}|} \\sum_{C \\in \\mathcal{C}}
          \\sum_{1 \\le i < j \\le k}
          \\frac{\\text{vol}(C_i \\cap C_j)}{\\min(\\text{vol}(C_i), \\text{vol}(C_j))},

    where :math:`\\text{vol}(C_i) = \\prod_{m=1}^{d} (\\text{softplus}(2|o_{C,i,m}|) + \\varepsilon)`
    and :math:`\\text{vol}(C_i \\cap C_j) = \\prod_{m=1}^{d} \\text{relu}(\\text{hi}_{ij,m} - \\text{lo}_{ij,m})`
    from exact corner arithmetic.

    Guarantees:
    - Returns a scalar ``torch.Tensor`` of shape ``()`` on ``centers.device``.
    - Returns ``0.0`` when ``k == 1`` (``n_components == 1``) or when ``centers``
      has zero classes.
    - Evaluates to ``0.0`` whenever all component pairs :math:`(C_i, C_j)` of
      every class are disjoint (or touch only on their boundary).
    - Strictly positive whenever any component pair shares an interior point
      (including nested or identical components with non-zero half-widths).

    Non-guarantees:
    - In high ambient dimensions :math:`d`, the coordinatewise product of side
      lengths scales exponentially with :math:`d`.

    Parameters
    ----------
    centers : torch.Tensor
        Box centres of shape ``(N, k, d)`` (or ``(k, d)`` for a single class).
    offsets : torch.Tensor
        Box half-widths of shape ``(N, k, d)`` (or ``(k, d)``); ``torch.abs`` is
        applied internally.
    eps : float, default 1e-6
        Small positive constant added to each softplus-clamped side length.

    Returns
    -------
    torch.Tensor
        Scalar diversity penalty averaged over classes and summed over pairs :math:`i < j`.
    """
    if centers.shape != offsets.shape:
        raise ValueError(
            f"centers and offsets must have the same shape, got {centers.shape} vs {offsets.shape}"
        )
    if centers.dim() == 2:
        centers = centers.unsqueeze(0)
        offsets = offsets.unsqueeze(0)
    elif centers.dim() != 3:
        raise ValueError(
            f"centers must have shape (N, k, d) or (k, d), got {centers.shape}"
        )

    n_classes, k, _ = centers.shape
    if n_classes == 0 or k < 2:
        return th.zeros((), device=centers.device, dtype=centers.dtype)

    half = th.abs(offsets)
    lo = centers - half  # (N, k, d)
    hi = centers + half  # (N, k, d)

    sides = 2.0 * half  # (N, k, d)
    vol = th.prod(F.softplus(sides) + eps, dim=-1)  # (N, k)

    idx_i, idx_j = th.triu_indices(k, k, offset=1, device=centers.device)
    inter_lo = th.maximum(lo[:, idx_i, :], lo[:, idx_j, :])  # (N, P, d)
    inter_hi = th.minimum(hi[:, idx_i, :], hi[:, idx_j, :])  # (N, P, d)
    inter_sides = th.relu(inter_hi - inter_lo)  # (N, P, d)
    inter_vol = th.prod(inter_sides, dim=-1)  # (N, P)

    min_vol = th.minimum(vol[:, idx_i], vol[:, idx_j])  # (N, P)
    ratio = inter_vol / th.clamp(min_vol, min=eps)  # (N, P)
    return ratio.sum(dim=1).mean(dim=0)
