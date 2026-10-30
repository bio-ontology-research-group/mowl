"""Soft component aggregation and diversity regularisation helpers.

Provides temperature-controlled soft minimum and softmax weights over component
axes so that every component of a multi-box class receives gradient during
training, a shared helper for soft GCI1 pairwise gating and inclusion, and a
log-space pairwise soft containment volume penalty that breaks collapsed or
nested component configurations without pushing apart components that are
already disjoint.

Guarantees:
- **Exact k = 1 identity:** When the aggregated dimension has size ``1``,
  :func:`softmin` returns the single slice (to floating-point precision),
  :func:`soft_weights` returns ``1.0``, and :func:`soft_gci1_terms` reduces to
  the single-box emptiness and inclusion terms for every positive temperature
  ``tau``.
- **Temperature limits:** For any tensor ``v``, :func:`softmin` converges to
  ``torch.amin(v, dim=dim)`` as :math:`\\tau \\to 0^+` (bounded within
  :math:`[\\min(v), \\min(v) + \\tau \\log K]`) and to ``v.mean(dim=dim)`` as
  :math:`\\tau \\to \\infty`.
- **Dimension-independent identical-box scaling of diversity penalty:** Computed
  in log space using raw side lengths :math:`2|o_{C,i,m}|` on both numerator and
  denominator, :func:`diversity_penalty` evaluates to ``1.0`` per identical
  component pair at any dimension :math:`d` (with non-zero gradient with respect
  to a centre shift), to :math:`\\approx 10^{-6}` per disjoint coordinate (not
  exact ``0.0``, and ``<= 1e-3`` per pair disjoint in at least one coordinate
  for side lengths :math:`\\ge 10^{-3}` at ``eps = 1e-6``), and to ``0.0`` when
  ``n_components == 1``.

Non-guarantees:
- **Non-zero soft loss at hard solutions and collapse pressure:** For finite
  :math:`\\tau > 0` and :math:`K > 1`, :func:`softmin` satisfies
  :math:`\\min(v) \\le \\text{softmin}_\\tau(v) \\le \\text{mean}(v)` and is strictly
  greater than :math:`\\min(v)` whenever the entries of :math:`v` are not all
  equal (for example, :math:`\\text{softmin}_{\\tau=1}([0, 5]) \\approx 0.69`).
  Consequently, a hard zero-loss model (where each source component :math:`C_i`
  is contained in one target component :math:`D_{j^*(i)}` while other target
  components :math:`D_j` are distant) is *not* a zero of the soft objective:
  every target component :math:`D_j` is pulled toward :math:`C_i` in proportion
  to its softmax weight :math:`\\text{softmax}(-v / \\tau)_j`, which drives
  target components to collapse onto one another unless counterbalanced by
  :func:`diversity_penalty` (``diversity_reg > 0``).
- **Float32 softmax underflow at very small temperatures:** In ``float32``,
  :math:`\\exp(-\\Delta / \\tau)` underflows to ``0.0`` once the score gap
  :math:`\\Delta = v_j - \\min(v)` exceeds :math:`\\approx 100 \\tau` (e.g. gaps
  :math:`\\ge 1` at :math:`\\tau = 0.01`), which zeros out gradients on
  non-nearest components.
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import torch as th

__all__ = [
    "softmin",
    "soft_weights",
    "soft_gci1_terms",
    "diversity_penalty",
]


def _validate_tau(tau: Union[float, int, th.Tensor]) -> float:
    """Validate that temperature ``tau`` is a strictly positive finite scalar."""
    if isinstance(tau, th.Tensor):
        if tau.dtype == th.bool or tau.numel() != 1:
            raise ValueError(f"tau must be a positive scalar, got {tau!r}")
        tau_f = float(tau.detach().item())
    elif isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise ValueError(f"tau must be a positive number, got {tau!r}")
    else:
        tau_f = float(tau)
    if tau_f <= 0.0 or math.isnan(tau_f) or math.isinf(tau_f):
        raise ValueError(f"tau must be > 0 and finite, got {tau!r}")
    return tau_f


def softmin(
    v: th.Tensor,
    tau: Union[float, int, th.Tensor],
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

    Non-guarantees:
    - At finite ``tau > 0``, ``softmin(v, tau)`` is strictly positive even when
      ``min(v) == 0`` if any other component has ``v_j > 0``; minimising a loss
      built from ``softmin`` therefore pulls all target components toward each
      source component (use :func:`diversity_penalty` via ``diversity_reg`` to
      counteract component collapse).
    - In ``float32``, gradient weights ``softmax(-v / tau)`` underflow to
      ``0.0`` when ``v_j - min(v)`` exceeds ``~100 * tau``.
    - Raises :class:`ValueError` if ``tau <= 0``.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor of scores.
    tau : float or torch.Tensor
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
    tau: Union[float, int, th.Tensor],
    dim: int = -1,
) -> th.Tensor:
    """Compute soft-argmin weights ``softmax(-v / tau, dim=dim)``.

    Guarantees:
    - Output has the same shape as ``v``, with non-negative entries that sum to
      ``1`` along ``dim``.
    - When ``v.shape[dim] == 1``, returns a tensor of ones with the same shape
      as ``v``.
    - Concentrates weight on ``argmin(v, dim=dim)`` as ``tau -> 0+`` and
      approaches uniform ``1 / K`` weights as ``tau -> inf``.

    Non-guarantees:
    - In ``float32``, entries whose score gap ``v_j - min(v)`` exceeds
      ``~100 * tau`` underflow to ``0.0``.
    - Raises :class:`ValueError` if ``tau <= 0``.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor of scores.
    tau : float or torch.Tensor
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


def soft_gci1_terms(
    empt: th.Tensor,
    non_empty: th.Tensor,
    incl_ijm: th.Tensor,
    tau: Union[float, int, th.Tensor],
) -> Tuple[th.Tensor, th.Tensor]:
    """Compute the soft GCI1 emptiness and gated inclusion terms for :math:`C \\sqcap D \\sqsubseteq E`.

    Given pairwise intersection emptiness scores :math:`\\text{empt}_{ij}` of
    shape ``(B, k, k)``, the non-empty intersection indicator
    :math:`\\text{non\\_empty}_{ij}` of shape ``(B, k, k)``, and pairwise-to-target
    inclusion scores :math:`\\text{incl}_{ijm}` of shape ``(B, k, k, k)``:

    - Computes soft-argmin weights
      :math:`w_{ij} = \\text{softmax}(-\\text{empt}_{ij} / \\tau)` over the
      :math:`k^2` pairs ``(i, j)``.
    - Keeps the gradient through :math:`w_{ij}` in the emptiness term
      :math:`\\text{empt\\_term} = \\sum_{i,j} w_{ij} \\,\\text{empt}_{ij}` so that
      when all component pairs are disjoint, every pair receives gradient pulling
      :math:`C_i` and :math:`D_j` toward intersection.
    - Detaches the gate
      :math:`g_{ij} = \\max(\\text{non\\_empty}_{ij}^{\\text{det}},\\, w_{ij}^{\\text{det}})`
      in the normalised inclusion term
      :math:`\\text{gated\\_incl} = \\frac{\\sum_{i,j} g_{ij}\\,\\text{softmin}_{\\tau,m}(\\text{incl}_{ijm})}{\\sum_{i,j} g_{ij}}`.
      Because the gate only selects which pairwise intersections are active and
      is not itself an optimisation target, detaching :math:`w_{ij}` in
      :math:`g_{ij}` prevents the normalised ratio from incentivising high-inclusion
      pairs to increase :math:`\\text{empt}_{ij}` (separating :math:`C_i` and
      :math:`D_j` instead of covering :math:`C_i \\cap D_j` with :math:`E_m`).

    Guarantees:
    - Returns ``(empt_term, gated_incl)``, each of shape ``(B,)``.
    - At ``k = 1``, ``w_00 = 1`` and ``g_00 = 1``, so ``empt_term = empt[:, 0, 0]``
      and ``gated_incl = incl_ijm[:, 0, 0, 0]`` identically for every ``tau > 0``.
    - ``gated_incl`` carries zero gradient with respect to ``empt``.

    Non-guarantees:
    - At finite ``tau > 0``, ``softmin`` over target components :math:`E_m` is
      strictly positive whenever the :math:`E_m` do not all yield zero inclusion
      error, pulling all :math:`E_m` toward each active intersection until
      counterbalanced by :func:`diversity_penalty`.

    Parameters
    ----------
    empt : torch.Tensor
        Pairwise emptiness penalties of shape ``(B, k, k)``.
    non_empty : torch.Tensor
        Boolean or float tensor of shape ``(B, k, k)`` indicating non-empty
        pairwise intersections; detached internally.
    incl_ijm : torch.Tensor
        Inclusion scores of shape ``(B, k, k, k)`` for each intersection pair
        ``(i, j)`` and target component ``m``.
    tau : float or torch.Tensor
        Strictly positive temperature :math:`\\tau > 0`.

    Returns
    -------
    empt_term : torch.Tensor
        Weighted emptiness penalty of shape ``(B,)``.
    gated_incl : torch.Tensor
        Detached-gate weighted mean of softmin inclusion scores of shape ``(B,)``.
    """
    tau_f = _validate_tau(tau)
    b_size, k, _ = empt.shape
    empt_flat = empt.reshape(b_size, k * k)

    w_flat = soft_weights(empt_flat, tau=tau_f, dim=1)  # (B, k * k)
    empt_term = (w_flat * empt_flat).sum(dim=1)  # (B,)

    w_det = w_flat.detach().view(b_size, k, k)  # (B, k, k)
    non_empty_det = non_empty.detach().to(dtype=empt.dtype)  # (B, k, k)
    gate = th.maximum(non_empty_det, w_det)  # (B, k, k)

    min_incl_ij = softmin(incl_ijm, tau=tau_f, dim=3)  # (B, k, k)
    gated_incl = (gate * min_incl_ij).sum(dim=(1, 2)) / gate.sum(dim=(1, 2))
    return empt_term, gated_incl


def diversity_penalty(
    centers: th.Tensor,
    offsets: th.Tensor,
    eps: float = 1e-6,
) -> th.Tensor:
    """Compute the log-space pairwise soft containment ratio across components of each class.

    For each class :math:`C` with :math:`k` box components
    :math:`C_i = [c_{C,i} - |o_{C,i}|,\\, c_{C,i} + |o_{C,i}|]`, raw coordinate
    side lengths :math:`\\text{side}_{i,m} = 2|o_{C,i,m}|`, and intersection side
    lengths :math:`\\text{inter}_{ij,m} = \\text{relu}(\\text{hi}_{ij,m} - \\text{lo}_{ij,m})`
    from corner arithmetic:

    .. math::

        \\text{ratio}_{ij}
        = \\exp\\!\\left(
            \\sum_{m=1}^{d} \\left[
              \\log(\\text{inter}_{ij,m} + \\varepsilon)
              - \\log(\\min(\\text{side}_{i,m}, \\text{side}_{j,m}) + \\varepsilon)
            \\right]
          \\right),

    .. math::

        \\mathcal{L}_{\\text{div}}
        = \\frac{1}{|\\mathcal{C}|} \\sum_{C \\in \\mathcal{C}}
          \\sum_{1 \\le i < j \\le k} \\text{ratio}_{ij}.

    Using the same raw side function :math:`2|o|` in both numerator and
    denominator and summing log-ratios before exponentiating ensures that
    identical or nested components yield :math:`\\text{ratio}_{ij} = 1` at any
    embedding dimension :math:`d` without exponential underflow, providing the
    repulsive counterweight needed when soft aggregation (:func:`softmin`) pulls
    target components together at finite temperature :math:`\\tau`.

    Guarantees:
    - Returns a scalar ``torch.Tensor`` of shape ``()`` on ``centers.device``.
    - Returns ``0.0`` when ``k == 1`` (``n_components == 1``) or when ``centers``
      has zero classes.
    - For two identical boxes (:math:`c_i = c_j,\\, |o_i| = |o_j| > 0`),
      :math:`\\text{ratio}_{ij} = 1` at any ambient dimension :math:`d` with a
      non-zero subgradient with respect to a centre shift (ties on lower and
      upper corners are broken asymmetrically so :math:`\\nabla_{c_i} \\text{ratio}_{ij} \\neq 0`
      even at exact coincidence :math:`c_i = c_j`).
    - When a smaller box :math:`C_i` is contained in or overlaps a larger box
      :math:`C_j`, :math:`\\text{ratio}_{ij}` equals the coordinatewise product
      :math:`\\prod_m (\\text{inter}_{ij,m} + \\varepsilon) / (\\min(\\text{side}_{i,m}, \\text{side}_{j,m}) + \\varepsilon)`.
    - When two boxes are disjoint in at least one coordinate (so
      :math:`\\text{inter}_{ij,m} = 0` in that coordinate), each disjoint
      coordinate contributes a factor of
      :math:`\\varepsilon / (\\min(\\text{side}_{i,m}, \\text{side}_{j,m}) + \\varepsilon) \\approx 10^{-6} / \\text{side}_{ij,m}`
      at ``eps = 1e-6`` (giving :math:`\\approx 10^{-6}` per disjoint coordinate
      for unit-scale boxes and :math:`\\text{ratio}_{ij} \\le 10^{-3}` for side
      lengths :math:`\\ge 10^{-3}`).

    Non-guarantees:
    - Because ``eps > 0`` is added inside :math:`\\log(\\text{inter}_{ij,m} + \\varepsilon)`
      to keep log-space evaluation finite, disjoint pairs contribute
      :math:`\\approx 10^{-6} / \\text{side}` per disjoint coordinate (:math:`O(\\varepsilon)`
      per disjoint coordinate, not exact IEEE ``0.0``, until the product across
      many disjoint coordinates underflows ``float32``).

    Parameters
    ----------
    centers : torch.Tensor
        Box centres of shape ``(N, k, d)`` (or ``(k, d)`` for a single class).
    offsets : torch.Tensor
        Box half-widths of shape ``(N, k, d)`` (or ``(k, d)``); ``torch.abs`` is
        applied internally.
    eps : float, default 1e-6
        Small positive constant added inside both logarithms for numerical
        stability.

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

    idx_i, idx_j = th.triu_indices(k, k, offset=1, device=centers.device)
    lo_i = lo[:, idx_i, :]  # (N, P, d)
    lo_j = lo[:, idx_j, :]  # (N, P, d)
    hi_i = hi[:, idx_i, :]  # (N, P, d)
    hi_j = hi[:, idx_j, :]  # (N, P, d)

    # Asymmetric tie-breaking on exact corner coincidence preserves the exact
    # forward corner arithmetic while selecting the subgradient (hi_j - lo_i)
    # at c_i == c_j, o_i == o_j so identical boxes have non-zero centre gradient.
    inter_lo = th.where(lo_i >= lo_j, lo_i, lo_j)  # (N, P, d)
    inter_hi = th.where(hi_i < hi_j, hi_i, hi_j)  # (N, P, d)
    inter_sides = th.relu(inter_hi - inter_lo)  # (N, P, d)

    min_sides = th.minimum(sides[:, idx_i, :], sides[:, idx_j, :])  # (N, P, d)
    log_ratio = (th.log(inter_sides + eps) - th.log(min_sides + eps)).sum(dim=-1)  # (N, P)
    ratio = th.exp(log_ratio)  # (N, P)
    return ratio.sum(dim=1).mean(dim=0)
