"""Backbone-agnostic point-based membership and loss functions for Objective O3.

Implements soft coordinatewise box membership, noisy-OR union membership across
multi-box components, binary cross-entropy membership loss, and log-side-length
volume regularisation.

Guarantees:
- :func:`soft_box_membership` returns a tensor in ``(0, 1)`` of shape ``(B, k)``
  computed as :math:`\\min_{\\text{coord}} \\sigma((|o| - |x - c|) / \\beta)`.
  The minimum over coordinates is used instead of the coordinatewise product
  because the product decays as :math:`\\sigma(|o|/\\beta)^d` even for a point at
  the box centre (e.g. :math:`\\approx 10^{-7}` at :math:`d=50, |o|=\\beta`),
  making membership unreachable in high ambient dimensions, whereas the minimum
  has no :math:`d`-dependence and still approaches ``1`` only when every
  coordinate lies inside the box.
- :func:`union_membership` computes the noisy-OR :math:`1 - \\prod_{i=1}^k (1 - m_i)`
  of shape ``(B,)``, reducing0-for-0 to ``m[:, 0]`` when ``k == 1``.
- :func:`membership_bce` clamps probabilities to ``[eps, 1 - eps]`` (``eps = 1e-6``)
  so the loss and its gradients are always finite even when ``m`` underflows to ``0``
  or saturates to ``1``.
- :func:`volume_penalty` returns the mean over classes and components of
  :math:`\\text{mean}_{\\text{coord}} \\log(2|o| + 10^{-6})`.

Non-guarantees:
- At finite :math:`\\beta > 0`, :func:`soft_box_membership` satisfies
  :math:`0 < m(x) < 1` strictly for finite box offsets.
- Consequently, :func:`membership_bce` is strictly positive and never reaches
  exact zero.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional, Union

import torch as th
import torch.nn as nn

if TYPE_CHECKING:
    from mowl.datasets.el.points import PointSupervision


_VALID_OBJECTIVES = ("axiom", "point", "mixed")


def _validate_beta(beta: float) -> float:
    """Validate that ``beta`` is a positive real number."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) <= 0.0:
        raise ValueError(f"beta must be a positive float, got {beta!r}")
    return float(beta)


def _validate_ind_halfwidth(ind_halfwidth: float) -> float:
    """Validate that ``ind_halfwidth`` is a positive real number."""
    if (
        isinstance(ind_halfwidth, bool)
        or not isinstance(ind_halfwidth, (int, float))
        or float(ind_halfwidth) <= 0.0
    ):
        raise ValueError(
            f"ind_halfwidth must be a positive float, got {ind_halfwidth!r}"
        )
    return float(ind_halfwidth)


def _validate_objective(objective: str) -> str:
    """Validate that ``objective`` is one of ``'axiom'``, ``'point'``, or ``'mixed'``."""
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"objective must be 'axiom', 'point', or 'mixed', got {objective!r}"
        )
    return objective


def soft_box_membership(
    x: th.Tensor,
    centers: th.Tensor,
    offsets: th.Tensor,
    beta: float = 0.1,
) -> th.Tensor:
    """Compute soft box membership :math:`m(x; c, o) = \\min_{\\text{coord}} \\sigma((|o| - |x - c|) / \\beta)`.

    Uses the minimum over coordinates rather than the coordinatewise product because
    the product decays exponentially with dimension as :math:`\\sigma(|o|/\\beta)^d`
    even at :math:`x = c`, whereas the minimum is dimension-free and equals :math:`1`
    in the :math:`\\beta \\to 0` limit if and only if every coordinate of :math:`x`
    lies inside :math:`[c - |o|, c + |o|]`.

    Guarantees:
    - Returns a tensor of shape ``(B, k)`` with entries strictly in ``(0, 1)``.
    - Dimension-invariant for isotropic boxes when :math:`x = c`.

    Non-guarantees:
    - Subgradients pass only through the bottleneck coordinate(s) achieving the
      minimum in each component box.

    Parameters
    ----------
    x : torch.Tensor
        Point coordinates of shape ``(B, d)`` or ``(B, 1, d)``.
    centers : torch.Tensor
        Component box centres of shape ``(B, k, d)`` (or ``(B, d)``, treated as ``k = 1``).
    offsets : torch.Tensor
        Component box half-widths of shape ``(B, k, d)`` (or ``(B, d)``, treated as ``k = 1``).
    beta : float, default 0.1
        Sigmoid temperature :math:`\\beta > 0`.

    Returns
    -------
    torch.Tensor
        Per-component soft membership values in ``(0, 1)`` of shape ``(B, k)``.
    """
    beta_f = _validate_beta(beta)
    if centers.dim() == 2:
        centers = centers.unsqueeze(1)
    if offsets.dim() == 2:
        offsets = offsets.unsqueeze(1)
    if x.dim() == 2:
        x_exp = x.unsqueeze(1)
    elif x.dim() == 3:
        x_exp = x
    else:
        raise ValueError(f"Expected x to have 2 or 3 dimensions, got shape {tuple(x.shape)}")

    logits = (th.abs(offsets) - th.abs(x_exp - centers)) / beta_f
    return th.amin(th.sigmoid(logits), dim=-1)


def union_membership(m: th.Tensor) -> th.Tensor:
    """Aggregate component memberships ``(B, k)`` via noisy-OR :math:`1 - \\prod_i (1 - m_i)`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` in ``[0, 1]``.
    - Reduces identically to ``m[:, 0]`` when ``k == 1``.

    Non-guarantees:
    - Assumes independent component memberships when combining overlapping boxes.

    Parameters
    ----------
    m : torch.Tensor
        Component memberships of shape ``(B, k)`` with entries in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Union membership probabilities of shape ``(B,)``.
    """
    if m.dim() != 2:
        raise ValueError(f"Expected m to have shape (B, k), got {tuple(m.shape)}")
    return 1.0 - th.prod(1.0 - m, dim=-1)


def membership_bce(
    m: th.Tensor,
    target: Union[float, int, th.Tensor],
    weight: Union[float, int, th.Tensor] = 1.0,
    eps: float = 1e-6,
) -> th.Tensor:
    """Compute weighted binary cross-entropy on membership probabilities ``m``.

    Guarantees:
    - Clamps ``m`` to ``[eps, 1 - eps]`` so log-probabilities and gradients are
      always finite.
    - Returns a scalar ``0.0`` tensor when ``m`` is empty.

    Non-guarantees:
    - Never reaches exact ``0.0`` for non-empty batches because ``m`` is clamped
      away from ``0`` and ``1``.

    Parameters
    ----------
    m : torch.Tensor
        Membership probabilities of shape ``(B,)``.
    target : float, int, or torch.Tensor
        Binary targets (``1.0`` for positive pairs, ``0.0`` for negative pairs),
        scalar or broadcastable to ``m``.
    weight : float, int, or torch.Tensor, default 1.0
        Sample weight multiplying the BCE loss, scalar or broadcastable to ``m``.
    eps : float, default 1e-6
        Clamping epsilon so ``m`` stays in ``[eps, 1 - eps]``.

    Returns
    -------
    torch.Tensor
        Scalar mean weighted BCE loss tensor.
    """
    if m.numel() == 0:
        return th.zeros((), device=m.device, dtype=m.dtype)
    m_clamped = m.clamp(min=eps, max=1.0 - eps)
    t = th.as_tensor(target, dtype=m.dtype, device=m.device).expand_as(m_clamped)
    w = th.as_tensor(weight, dtype=m.dtype, device=m.device).expand_as(m_clamped)
    bce = -(t * th.log(m_clamped) + (1.0 - t) * th.log(1.0 - m_clamped)) * w
    return bce.mean()


def volume_penalty(offsets: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    """Compute mean log side length :math:`\\text{mean}_{C, i} \\text{mean}_{\\text{coord}} \\log(2|o| + 10^{-6})`.

    Guarantees:
    - Returns a finite scalar tensor for any non-negative or signed ``offsets``.
    - Returns ``0.0`` when ``offsets`` is empty.

    Non-guarantees:
    - Can take negative values when box side lengths :math:`2|o|` are smaller than ``1``.

    Parameters
    ----------
    offsets : torch.Tensor
        Box half-widths of shape ``(C, k, d)`` or ``(C, k * d)``.
    eps : float, default 1e-6
        Small positive additive constant inside the logarithm.

    Returns
    -------
    torch.Tensor
        Scalar volume penalty tensor.
    """
    if offsets.numel() == 0:
        return th.zeros((), device=offsets.device, dtype=offsets.dtype)
    return th.log(2.0 * th.abs(offsets) + eps).mean(dim=-1).mean()


def gather_element_embeddings(
    elem_idx: th.Tensor,
    ps: PointSupervision,
    elem_embed: nn.Embedding,
    ind_embed: Optional[nn.Embedding],
) -> th.Tensor:
    """Look up coordinates for element indices ``elem_idx``, aliasing individual elements to ``ind_embed``.

    Elements with ``ps.element_is_individual[e] == True`` are gathered from
    ``ind_embed(ps.element_class_index[e])``; class elements are gathered from
    ``elem_embed(e)``.

    Guarantees:
    - Gradients for individual elements route exclusively to ``ind_embed.weight``
      (``elem_embed.weight`` is not indexed at individual element rows).
    - Gradients for class elements route exclusively to ``elem_embed.weight``.

    Non-guarantees:
    - Raises :class:`ValueError` if ``elem_idx`` references an individual element
      when ``ind_embed is None``.

    Parameters
    ----------
    elem_idx : torch.Tensor
        1-D integer tensor of canonical element indices of shape ``(B,)``.
    ps : PointSupervision
        Canonical-element supervision structure defining element-to-entity mappings.
    elem_embed : torch.nn.Embedding
        Element embedding table of shape ``(n_elements, d)``.
    ind_embed : torch.nn.Embedding, optional
        Individual embedding table of shape ``(n_individuals, d)``, or ``None``.

    Returns
    -------
    torch.Tensor
        Element point coordinates of shape ``(B, d)``.
    """
    device = elem_embed.weight.device
    dtype = elem_embed.weight.dtype
    idx = elem_idx.to(device=device, dtype=th.long)
    if idx.numel() == 0:
        return th.empty((0, elem_embed.embedding_dim), device=device, dtype=dtype)

    is_ind = ps.element_is_individual.to(device=device)[idx]
    if not bool(is_ind.any().item()):
        return elem_embed(idx)

    if ind_embed is None:
        raise ValueError(
            "PointSupervision contains individual elements, but module.ind_embed is None."
        )
    cls_or_ind = ps.element_class_index.to(device=device)[idx]
    if bool(is_ind.all().item()):
        return ind_embed(cls_or_ind)

    out = th.empty(
        (idx.shape[0], elem_embed.embedding_dim), device=device, dtype=dtype
    )
    out[~is_ind] = elem_embed(idx[~is_ind])
    out[is_ind] = ind_embed(cls_or_ind[is_ind])
    return out


def pair_union_membership(
    module: nn.Module,
    pairs: th.Tensor,
) -> th.Tensor:
    """Compute union membership :math:`m_C(x_D)` for a ``(B, 2)`` tensor of ``[elem_idx, class_idx]``.

    Guarantees:
    - Evaluates :func:`soft_box_membership` at temperature ``module.beta`` and
      aggregates across the ``k`` components of ``class_idx`` via :func:`union_membership`.
    - Returns an empty 1-D tensor on ``module``'s device when ``pairs`` is empty.

    Non-guarantees:
    - Requires ``module.set_point_supervision(ps)`` to have been called prior to invocation.

    Parameters
    ----------
    module : torch.nn.Module
        Multi-box module exposing ``element_embeddings``, ``class_boxes``, and ``beta``.
    pairs : torch.Tensor
        2-D integer tensor of shape ``(B, 2)`` with columns ``[elem_idx, class_idx]``.

    Returns
    -------
    torch.Tensor
        1-D tensor of union membership probabilities of shape ``(B,)``.
    """
    device = module.class_center.weight.device  # type: ignore[union-attr]
    dtype = module.class_center.weight.dtype  # type: ignore[union-attr]
    if pairs.numel() == 0:
        return th.empty((0,), device=device, dtype=dtype)
    p = pairs.to(device=device, dtype=th.long)
    x = module.element_embeddings(p[:, 0])  # type: ignore[operator]
    cen, off = module.class_boxes(p[:, 1])  # type: ignore[operator]
    return union_membership(
        soft_box_membership(x, cen, off, beta=float(module.beta))  # type: ignore[arg-type]
    )


def compute_point_membership_loss(
    module: nn.Module,
    batch: Union[Mapping[str, th.Tensor], th.Tensor],
    neg: bool = False,
    weight: Optional[float] = None,
) -> th.Tensor:
    """Compute Objective O3 membership BCE loss for a sampled dict or a ``(B, 2)`` pair tensor.

    Guarantees:
    - When ``batch`` is a mapping, sums the BCE losses for ``"pos"`` (target ``1``,
      weight ``1``), ``"hard"`` (target ``0``, weight ``1``), and ``"rand"``
      (target ``0``, weight ``weight`` or ``module.neg_weight``).
    - When ``batch`` is a ``(B, 2)`` tensor, computes BCE with target ``0`` if
      ``neg=True`` else ``1``.

    Non-guarantees:
    - Ignores role triples ``"roles"`` (handled separately by ``point_role_loss``).

    Parameters
    ----------
    module : torch.nn.Module
        Multi-box module with attached :class:`~mowl.datasets.el.points.PointSupervision`.
    batch : Mapping[str, torch.Tensor] or torch.Tensor
        Dictionary returned by :meth:`~mowl.datasets.el.points.PointSupervision.sample`
        or a 2-D pair tensor of shape ``(B, 2)``.
    neg : bool, default False
        Whether a tensor ``batch`` represents negative pairs.
    weight : float, optional
        Optional override for the negative-sample weight.

    Returns
    -------
    torch.Tensor
        Scalar BCE loss tensor.
    """
    device = module.class_center.weight.device  # type: ignore[union-attr]
    dtype = module.class_center.weight.dtype  # type: ignore[union-attr]

    if isinstance(batch, Mapping):
        loss = th.zeros((), device=device, dtype=dtype)
        pos = batch.get("pos")
        if pos is not None and pos.numel() > 0:
            m_pos = pair_union_membership(module, pos)
            loss = loss + membership_bce(m_pos, 1.0, weight=1.0)
        hard = batch.get("hard")
        if hard is not None and hard.numel() > 0:
            m_hard = pair_union_membership(module, hard)
            loss = loss + membership_bce(m_hard, 0.0, weight=1.0)
        rand = batch.get("rand")
        neg_w = float(getattr(module, "neg_weight", 0.1)) if weight is None else float(weight)
        if rand is not None and rand.numel() > 0 and neg_w > 0.0:
            m_rand = pair_union_membership(module, rand)
            loss = loss + membership_bce(m_rand, 0.0, weight=neg_w)
        return loss

    m = pair_union_membership(module, batch)
    target = 0.0 if neg else 1.0
    eff_w = (float(getattr(module, "neg_weight", 0.1)) if neg else 1.0) if weight is None else float(weight)
    return membership_bce(m, target, weight=eff_w)


def compute_class_assertion_score_point(
    module: nn.Module,
    data: th.Tensor,
    eps: float = 1e-12,
) -> th.Tensor:
    """Compute point-mode class-assertion ranking score :math:`-\\log m_C(a)` for ``data`` ``(B, 2)`` ``[ind, class]``.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` where smaller values indicate higher
      membership probability :math:`m_C(a)`.
    - Reads individual coordinates exclusively from ``module.ind_embed`` without
      accessing ``ind_offset``.

    Non-guarantees:
    - Raises :class:`ValueError` when the module has no individuals (``nb_inds`` is
      ``None`` or ``0``).

    Parameters
    ----------
    module : torch.nn.Module
        Multi-box module exposing ``ind_embed``, ``class_boxes``, and ``beta``.
    data : torch.Tensor
        2-D integer tensor of shape ``(B, 2)`` with columns ``[ind_idx, class_idx]``.
    eps : float, default 1e-12
        Lower clamp on :math:`m_C(a)` inside ``-log``.

    Returns
    -------
    torch.Tensor
        1-D score tensor of shape ``(B,)``.
    """
    ind_embed = getattr(module, "ind_embed", None)
    nb_inds = getattr(module, "nb_inds", None)
    if ind_embed is None or nb_inds is None or int(nb_inds) == 0:
        raise ValueError(
            "The number of individuals must be specified to use this loss function."
        )
    device = module.class_center.weight.device  # type: ignore[union-attr]
    d_tensor = data.to(device=device, dtype=th.long)
    x = ind_embed(d_tensor[:, 0])
    cen, off = module.class_boxes(d_tensor[:, 1])  # type: ignore[operator]
    m = union_membership(
        soft_box_membership(x, cen, off, beta=float(module.beta))  # type: ignore[arg-type]
    )
    return -th.log(m.clamp(min=eps))


def compute_point_modelness(
    module: nn.Module,
    ps: Optional[PointSupervision] = None,
    chunk_size: int = 8192,
) -> Dict[str, float]:
    """Compute fraction of positives with ``m >= 0.5`` and hard negatives with ``m < 0.5`` in chunks.

    Guarantees:
    - Evaluates all pairs in ``ps.pos_pairs`` and ``ps.hard_neg_pairs`` on
      ``module``'s current device under ``torch.no_grad()`` using batches of size
      at most ``chunk_size``.
    - Returns ``1.0`` for ``pos_frac`` (or ``hardneg_frac``) when the corresponding
      pair tensor is empty.

    Non-guarantees:
    - Does not evaluate role pairs or random negatives.

    Parameters
    ----------
    module : torch.nn.Module
        Multi-box module to evaluate.
    ps : PointSupervision, optional
        Supervision object; defaults to ``module.point_supervision``.
    chunk_size : int, default 8192
        Maximum number of pairs evaluated per forward chunk.

    Returns
    -------
    Dict[str, float]
        Dictionary ``{"pos_frac": float, "hardneg_frac": float}``.
    """
    eff_ps = ps if ps is not None else getattr(module, "point_supervision", None)
    if eff_ps is None:
        raise ValueError(
            "PointSupervision must be provided or set via set_point_supervision(ps)."
        )
    if getattr(module, "point_supervision", None) is None:
        module.set_point_supervision(eff_ps)  # type: ignore[operator]

    step = max(int(chunk_size), 1)
    with th.no_grad():
        n_pos = int(eff_ps.pos_pairs.shape[0])
        if n_pos == 0:
            pos_frac = 1.0
        else:
            pos_ok = 0
            for start in range(0, n_pos, step):
                chunk = eff_ps.pos_pairs[start : start + step]
                m = pair_union_membership(module, chunk)
                pos_ok += int((m >= 0.5).sum().item())
            pos_frac = float(pos_ok / n_pos)

        n_hard = int(eff_ps.hard_neg_pairs.shape[0])
        if n_hard == 0:
            hardneg_frac = 1.0
        else:
            hard_ok = 0
            for start in range(0, n_hard, step):
                chunk = eff_ps.hard_neg_pairs[start : start + step]
                m = pair_union_membership(module, chunk)
                hard_ok += int((m < 0.5).sum().item())
            hardneg_frac = float(hard_ok / n_hard)

    return {"pos_frac": pos_frac, "hardneg_frac": hardneg_frac}

