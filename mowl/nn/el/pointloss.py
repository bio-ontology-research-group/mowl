"""Backbone-agnostic point-based membership and loss functions for Objective O3.

Implements soft coordinatewise box membership, noisy-OR union membership across
multi-box components, binary cross-entropy membership loss, and log-side-length
volume regularisation.

Guarantees:
- :func:`soft_box_membership` returns a tensor in ``(0, 1)`` of shape ``(B, k)``
  computed as :math:`\\prod_{\\text{coord}} \\sigma((|o| - |x - c|) / \\beta)`.
- :func:`union_membership` computes the noisy-OR :math:`1 - \\prod_{i=1}^k (1 - m_i)`
  of shape ``(B,)``, reducing exactly to ``m[:, 0]`` when ``k == 1``.
- :func:`membership_bce` clamps probabilities to ``[eps, 1 - eps]`` (``eps = 1e-6``)
  so the loss and its gradients are always finite even when ``m`` underflows to ``0``
  or saturates to ``1``.
- :func:`volume_penalty` returns the mean over classes and components of
  :math:`\\text{mean}_{\\text{coord}} \\log(2|o| + 10^{-6})`.

Non-guarantees:
- Because :func:`soft_box_membership` is a product of sigmoids over ``d``
  coordinates, ``m(x) < 1`` strictly at finite ``beta > 0`` and decreases with
  ambient dimension ``d`` unless ``|o| - |x - c|`` is large relative to ``beta``.
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
    """Compute soft box membership :math:`m(x; c, o) = \\prod_{\\text{coord}} \\sigma((|o| - |x - c|) / \\beta)`.

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
    return th.prod(th.sigmoid(logits), dim=-1)


def union_membership(m: th.Tensor) -> th.Tensor:
    """Aggregate component memberships ``(B, k)`` via noisy-OR :math:`1 - \\prod_i (1 - m_i)`.

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
    """Compute union membership :math:`m_C(x_D)` for a ``(B, 2)`` tensor of ``[elem_idx, class_idx]``."""
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
    """Compute Objective O3 membership BCE loss for a sampled dict or a ``(B, 2)`` pair tensor."""
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
    """Compute point-mode class-assertion ranking score :math:`-\\log m_C(a)` for ``data`` ``(B, 2)`` ``[ind, class]``."""
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
    """Compute fraction of positives with ``m >= 0.5`` and hard negatives with ``m < 0.5`` in chunks."""
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

