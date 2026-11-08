"""Query2Box / BetaE query compiler and region scorers for GroundedEL in translation mode.

Implements §4 of ``2026-11-08-query-answering-design.md``:

1. **Batched multi-box representation (:class:`MultiBox`):** Represents a query
   region for a batch of :math:`B` queries as up to :math:`M` axis-aligned boxes
   with centres ``cen`` of shape ``(B, M, d)``, non-negative half-widths ``w``
   of shape ``(B, M, d)``, and boolean mask ``active`` of shape ``(B, M)``.
2. **Region algebra (:func:`anchor`, :func:`project`, :func:`intersect`,
   :func:`union`, :func:`class_boxes`, :func:`project_exact`):** Compiles
   constants to width-0 point boxes, role projections to Minkowski sums
   :math:`(c_m + c_{r,k},\\, w_m + w_{r,k})`, conjunctions to pairwise box
   intersections with empty intersections dropped, and disjunctions to box
   concatenation, capping slot count at ``max_boxes`` by retaining the
   largest-volume active boxes.
3. **Depth and overlap measures (:func:`depth`, :func:`overlap_depth`,
   :func:`score_entities`, :func:`score_concepts`):** Evaluates signed point-in-region
   depth :math:`\\max_m \\min_j (w_{m,j} - |p_j - c_{m,j}|)` over candidate
   entities and pairwise box-overlap depth
   :math:`\\max_{m,q} \\min_j (w_{m,j} + w_{q,j} - |c_{m,j} - c_{q,j}|)` over
   candidate concepts.
4. **Recursive query compiler (:func:`compile_query`):** Compiles standard
   Query2Box structures (``1p``, ``2p``, ``3p``, ``2i``, ``3i``, ``2u``, ``ip``,
   ``pi``, ``up``) into a single :class:`MultiBox` batch, supporting dataset
   entity-to-element index mappings, special ``type_inv_rel`` / ``type_rel``
   relations, and ``exact=True`` element-witness projection after the first hop.

Guarantees:
- **Reasoner-free (R1):** Contains no imports or calls to reasoners or semantic
  completion routines.
- **Exact box algebra below cap:** Whenever the number of active boxes at every
  intermediate step is at most ``max_boxes``, ``score_entities(module, compile_query(...)) > 0``
  equals the exact Minkowski-sum and box-intersection region semantics, and
  ``compile_query(..., exact=True)`` with ``max_witnesses >= nb_elements`` equals
  the exact element-witness relational semantics after the first hop.
- **Differentiable and batch-vectorised:** All operations avoid ``.item()`` and
  Python loops over the query batch, remaining differentiable when gradients are
  enabled and ``no_grad``-safe during evaluation.

Non-guarantees:
- When a projection, intersection, union, or ``type_rel`` step produces more than
  ``max_boxes`` active boxes, only the ``max_boxes`` boxes of largest Euclidean
  volume are retained.
- In ``project_exact``, witness elements at each hop are selected by hard
  thresholding ``depth(p_z, R) > 0`` (top ``max_witnesses`` by depth), so
  gradients do not flow through the discrete witness selection step.
"""
from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from mowl.nn.el.grounded.module import GroundedELModule
from mowl.nn.el.structural import Regions

__all__ = [
    "MultiBox",
    "anchor",
    "project",
    "intersect",
    "union",
    "class_boxes",
    "depth",
    "overlap_depth",
    "compile_query",
    "score_entities",
    "score_concepts",
    "project_exact",
]

_EMPTY_DEPTH_FLOOR: float = -1e6
_TOP_DEPTH_CEIL: float = 1e6
_LSE_DEFAULT_TAU: float = 0.1
_LOG_VOL_FLOOR: float = 1e-30


class MultiBox(NamedTuple):
    """Batched union of axis-aligned boxes over a query batch.

    Guarantees:
    - Holds ``cen`` of shape ``(B, M, d)``, non-negative half-widths ``w`` of
      shape ``(B, M, d)``, and boolean mask ``active`` of shape ``(B, M)``.

    Non-guarantees:
    - Does not copy or validate tensors on construction; helper functions
      validate shapes and zero out inactive padding slots.

    Attributes
    ----------
    cen : torch.Tensor
        Box centres of shape ``(B, M, d)``.
    w : torch.Tensor
        Box half-widths of shape ``(B, M, d)`` with ``w >= 0``.
    active : torch.Tensor
        Boolean mask of shape ``(B, M)`` where ``False`` marks inactive padding slots.
    """

    cen: th.Tensor
    w: th.Tensor
    active: th.Tensor


def _validate_int_ge(name: str, val: int, min_val: int) -> int:
    """Validate that ``val`` is an integer ``>= min_val`` (rejecting booleans)."""
    if isinstance(val, bool) or not isinstance(val, int) or val < min_val:
        raise ValueError(f"{name} must be an int >= {min_val}, got {val!r}")
    return int(val)


def _validate_module(module: GroundedELModule, require_translation: bool = False) -> None:
    """Validate that ``module`` is a :class:`GroundedELModule` (and optionally in translation mode)."""
    if not isinstance(module, GroundedELModule):
        raise TypeError(
            f"module must be a GroundedELModule, got {type(module).__name__}"
        )
    if require_translation and module.role_mode != "translation":
        raise ValueError(
            f"GroundedEL query compilation requires role_mode='translation', got {module.role_mode!r}"
        )


def _validate_multibox(
    R: Union[MultiBox, Regions],
    name: str = "R",
    expected_dim: Optional[int] = None,
) -> MultiBox:
    """Validate a :class:`MultiBox` (or :class:`Regions`) and return it as a :class:`MultiBox`."""
    if not isinstance(R, (MultiBox, Regions)):
        raise TypeError(f"{name} must be a MultiBox, got {type(R).__name__}")
    if (
        not isinstance(R.cen, th.Tensor)
        or not isinstance(R.w, th.Tensor)
        or not isinstance(R.active, th.Tensor)
    ):
        raise TypeError(f"{name} fields (cen, w, active) must be torch.Tensors")
    if (
        R.cen.ndim != 3
        or R.w.ndim != 3
        or R.active.ndim != 2
        or R.cen.shape != R.w.shape
        or R.cen.shape[:2] != R.active.shape
    ):
        raise ValueError(
            f"Invalid {name} shapes: cen={tuple(R.cen.shape)}, w={tuple(R.w.shape)}, "
            f"active={tuple(R.active.shape)}"
        )
    if R.active.dtype != th.bool:
        raise ValueError(f"{name}.active must have dtype torch.bool, got {R.active.dtype}")
    if expected_dim is not None and R.cen.shape[2] != expected_dim:
        raise ValueError(
            f"{name} embed_dim ({R.cen.shape[2]}) does not match module embed_dim ({expected_dim})"
        )
    if isinstance(R, MultiBox):
        return R
    return MultiBox(cen=R.cen, w=R.w, active=R.active)


def _to_1d_long(
    ids: Union[int, Sequence[int], np.ndarray, th.Tensor],
    device: th.device,
    name: str,
) -> th.Tensor:
    """Convert ``ids`` to a 1-D ``torch.long`` tensor on ``device`` without calling ``.item()``."""
    if isinstance(ids, (int, np.integer)) and not isinstance(ids, (bool, np.bool_)):
        return th.tensor([int(ids)], dtype=th.long, device=device)
    if isinstance(ids, th.Tensor):
        if ids.dtype == th.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError(f"{name} must contain integer indices, got dtype={ids.dtype}")
        idx = ids.to(device=device, dtype=th.long)
        if idx.ndim == 0:
            idx = idx.unsqueeze(0)
        elif idx.ndim != 1:
            raise ValueError(f"{name} must be 1-D (or scalar), got shape {tuple(idx.shape)}")
        return idx
    if isinstance(ids, (list, tuple, np.ndarray)):
        arr = np.asarray(ids)
        if arr.size > 0 and (
            arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer)
        ):
            raise ValueError(f"{name} must contain integer indices, got dtype={arr.dtype}")
        idx = th.as_tensor(arr, dtype=th.long, device=device)
        if idx.ndim == 0:
            idx = idx.unsqueeze(0)
        elif idx.ndim != 1:
            raise ValueError(f"{name} must be 1-D (or scalar), got shape {tuple(idx.shape)}")
        return idx
    raise TypeError(f"{name} must be an int, Sequence[int], or Tensor, got {type(ids).__name__}")


def _align_multibox_batch(R1: MultiBox, R2: MultiBox) -> Tuple[MultiBox, MultiBox]:
    """Validate and broadcast batch dimensions of ``R1`` and ``R2``."""
    if R1.cen.shape[2] != R2.cen.shape[2]:
        raise ValueError(
            f"Incompatible embedding dimensions: {R1.cen.shape[2]} vs {R2.cen.shape[2]}"
        )
    b1 = R1.cen.shape[0]
    b2 = R2.cen.shape[0]
    if b1 == b2:
        return R1, R2
    if b1 == 1 and b2 > 1:
        m1, d = R1.cen.shape[1], R1.cen.shape[2]
        return (
            MultiBox(
                cen=R1.cen.expand(b2, m1, d),
                w=R1.w.expand(b2, m1, d),
                active=R1.active.expand(b2, m1),
            ),
            R2,
        )
    if b2 == 1 and b1 > 1:
        m2, d = R2.cen.shape[1], R2.cen.shape[2]
        return (
            R1,
            MultiBox(
                cen=R2.cen.expand(b1, m2, d),
                w=R2.w.expand(b1, m2, d),
                active=R2.active.expand(b1, m2),
            ),
        )
    raise ValueError(f"Incompatible MultiBox batch sizes: {b1} vs {b2}")


def _compact_active_first(cen: th.Tensor, w: th.Tensor, active: th.Tensor) -> MultiBox:
    """Stably reorder slots so active boxes precede inactive boxes, zeroing inactive slots."""
    bsz, total, dim = cen.shape
    if total == 0 or bsz == 0:
        return MultiBox(cen=cen, w=th.clamp_min(w, 0.0), active=active)
    idx_grid = th.arange(total, device=active.device).unsqueeze(0)
    sort_key = th.where(active, idx_grid, idx_grid + total)
    order = th.argsort(sort_key, dim=1)
    order_d = order.unsqueeze(-1).expand(bsz, total, dim)
    cen_out = th.gather(cen, 1, order_d)
    w_out = th.gather(w, 1, order_d)
    act_out = th.gather(active, 1, order)
    act_mask = act_out.unsqueeze(-1)
    cen_out = th.where(act_mask, cen_out, th.zeros_like(cen_out))
    w_out = th.where(act_mask, th.clamp_min(w_out, 0.0), th.zeros_like(w_out))
    return MultiBox(cen=cen_out, w=w_out, active=act_out)


def _cap_boxes(R: MultiBox, max_boxes: int) -> MultiBox:
    """Cap ``R`` to at most ``max_boxes`` slots by keeping the largest-volume active boxes."""
    bsz, total, dim = R.cen.shape
    if total <= max_boxes:
        act_mask = R.active.unsqueeze(-1)
        cen_out = th.where(act_mask, R.cen, th.zeros_like(R.cen))
        w_out = th.where(act_mask, th.clamp_min(R.w, 0.0), th.zeros_like(R.w))
        return MultiBox(cen=cen_out, w=w_out, active=R.active)
    if bsz == 0:
        return MultiBox(
            cen=R.cen[:, :max_boxes, :],
            w=th.clamp_min(R.w[:, :max_boxes, :], 0.0),
            active=R.active[:, :max_boxes],
        )

    w_safe = th.clamp_min(R.w, _LOG_VOL_FLOOR)
    log_vol = th.log(w_safe).sum(dim=-1)  # (B, total)
    score = th.where(R.active, log_vol, th.full_like(log_vol, float("-inf")))
    _, top_idx = th.topk(score, k=max_boxes, dim=1, largest=True, sorted=True)
    idx_d = top_idx.unsqueeze(-1).expand(bsz, max_boxes, dim)
    cen_out = th.gather(R.cen, 1, idx_d)
    w_out = th.gather(R.w, 1, idx_d)
    act_out = th.gather(R.active, 1, top_idx)
    act_mask = act_out.unsqueeze(-1)
    cen_out = th.where(act_mask, cen_out, th.zeros_like(cen_out))
    w_out = th.where(act_mask, th.clamp_min(w_out, 0.0), th.zeros_like(w_out))
    return MultiBox(cen=cen_out, w=w_out, active=act_out)


def anchor(
    module: GroundedELModule,
    entity_element_ids: Union[int, Sequence[int], np.ndarray, th.Tensor],
) -> MultiBox:
    """Materialise width-0 point boxes at the positions of ``entity_element_ids``.

    Guarantees:
    - Returns a :class:`MultiBox` with ``cen`` of shape ``(B, 1, d)`` equal to
      ``module.positions(entity_element_ids).unsqueeze(1)``, ``w`` equal to
      exact ``0.0`` of shape ``(B, 1, d)``, and ``active`` equal to ``True`` of
      shape ``(B, 1)``.
    - Raises :class:`ValueError` if any index is outside ``[0, module.nb_elements)``.

    Non-guarantees:
    - Because half-widths are ``0.0``, strict interior depth ``depth(p, anchor(module, e))``
      is at most ``0.0`` prior to projecting along a role.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module holding element positions.
    entity_element_ids : Union[int, Sequence[int], numpy.ndarray, torch.Tensor]
        1-D element indices of shape ``(B,)`` (or scalar) in ``[0, module.nb_elements)``.

    Returns
    -------
    MultiBox
        Width-0 point boxes of shape ``(B, 1, d)``.
    """
    _validate_module(module, require_translation=False)
    device = module.elem_pos.weight.device
    idx = _to_1d_long(entity_element_ids, device, "entity_element_ids")
    pos = module.positions(idx)  # (B, d)
    cen = pos.unsqueeze(1)  # (B, 1, d)
    w = th.zeros_like(cen)
    active = th.ones((cen.shape[0], 1), dtype=th.bool, device=cen.device)
    return MultiBox(cen=cen, w=w, active=active)


def project(
    module: GroundedELModule,
    R: MultiBox,
    rel_ids: Union[int, Sequence[int], np.ndarray, th.Tensor],
    max_boxes: int = 64,
) -> MultiBox:
    """Project region ``R`` along translation role(s) ``rel_ids`` via Minkowski sum.

    Forms :math:`\\bigcup_{m, k} (c_m + c_{r,k},\\, w_m + w_{r,k})` over all active
    boxes :math:`m` of ``R`` and active translation components :math:`k` of ``rel_ids``,
    then caps to ``max_boxes`` by retaining the largest-volume active boxes.

    Guarantees:
    - Returns a :class:`MultiBox` of shape ``(B, min(M * K_r, max_boxes), d)``.
    - When ``M * K_r <= max_boxes``, represents the exact Minkowski sum
      :math:`R + V_r = \\{x + v : x \\in R,\\, v \\in V_r\\}`.
    - Raises :class:`ValueError` if ``module.role_mode != "translation"`` or
      ``max_boxes < 1``.

    Non-guarantees:
    - When ``M * K_r > max_boxes``, drops smaller-volume boxes to stay within
      ``max_boxes``.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module in ``"translation"`` role mode.
    R : MultiBox
        Input multi-box region of shape ``(B, M, d)``.
    rel_ids : Union[int, Sequence[int], numpy.ndarray, torch.Tensor]
        Role indices of shape ``(B,)`` (or scalar broadcast to ``B``) in
        ``[0, module.nb_rels)``.
    max_boxes : int, default 64
        Maximum number of box slots retained per query.

    Returns
    -------
    MultiBox
        Projected multi-box region of shape ``(B, M_out, d)``.
    """
    _validate_module(module, require_translation=True)
    R_val = _validate_multibox(R, "R", expected_dim=module.embed_dim)
    max_b = _validate_int_ge("max_boxes", max_boxes, 1)

    bsz, m_in, dim = R_val.cen.shape
    device = R_val.cen.device
    r_idx = _to_1d_long(rel_ids, device, "rel_ids")
    if r_idx.shape[0] == 1 and bsz > 1:
        r_idx = r_idx.expand(bsz)
    elif bsz == 0 and r_idx.shape[0] == 0:
        pass
    elif r_idx.shape[0] != bsz:
        raise ValueError(
            f"Incompatible batch sizes between R ({bsz}) and rel_ids ({r_idx.shape[0]})"
        )

    rel_reg = module.translation_regions(r_idx)
    kr = rel_reg.cen.shape[1]
    total = m_in * kr
    if total == 0:
        return MultiBox(
            cen=th.empty((bsz, 0, dim), dtype=R_val.cen.dtype, device=device),
            w=th.empty((bsz, 0, dim), dtype=R_val.w.dtype, device=device),
            active=th.empty((bsz, 0), dtype=th.bool, device=device),
        )

    r_cen = rel_reg.cen.to(dtype=R_val.cen.dtype)
    r_w = rel_reg.w.to(dtype=R_val.w.dtype)
    cen_mk = (R_val.cen.unsqueeze(2) + r_cen.unsqueeze(1)).reshape(bsz, total, dim)
    w_mk = (R_val.w.unsqueeze(2) + r_w.unsqueeze(1)).reshape(bsz, total, dim)
    act_mk = (R_val.active.unsqueeze(2) & rel_reg.active.unsqueeze(1)).reshape(bsz, total)

    act_mask = act_mk.unsqueeze(-1)
    cen_mk = th.where(act_mask, cen_mk, th.zeros_like(cen_mk))
    w_mk = th.where(act_mask, w_mk, th.zeros_like(w_mk))
    out = MultiBox(cen=cen_mk, w=w_mk, active=act_mk)
    if total > max_b:
        return _cap_boxes(out, max_b)
    return out


def intersect(
    R1: MultiBox,
    R2: MultiBox,
    max_boxes: int = 64,
) -> MultiBox:
    """Compute pairwise box intersections of ``R1`` and ``R2``, dropping empty intersections.

    For each pair :math:`(m_1, m_2)` of boxes from ``R1`` and ``R2``, forms
    :math:`\\text{lo} = \\max(c_1 - w_1,\\, c_2 - w_2)` and
    :math:`\\text{hi} = \\min(c_1 + w_1,\\, c_2 + w_2)`. A pair is active if and
    only if both parent boxes are active and :math:`\\text{hi}_j > \\text{lo}_j`
    in every coordinate :math:`j`.

    Guarantees:
    - Returns a :class:`MultiBox` of shape ``(B, min(M1 * M2, max_boxes), d)``
      with non-negative half-widths ``w >= 0``.
    - Empty pairwise intersections (:math:`\\min_j (\\text{hi}_j - \\text{lo}_j) \\le 0`)
      are marked ``active = False`` and placed after all active intersections
      (or dropped when capping to ``max_boxes``).
    - When the number of non-empty active intersections is at most ``max_boxes``,
      the active boxes exactly equal the non-empty pairwise intersections of
      ``R1`` and ``R2``.

    Non-guarantees:
    - When more than ``max_boxes`` pairwise intersections are simultaneously
      active and non-empty, only the ``max_boxes`` largest-volume intersections
      are retained.

    Parameters
    ----------
    R1 : MultiBox
        First multi-box region of shape ``(B, M1, d)``.
    R2 : MultiBox
        Second multi-box region of shape ``(B, M2, d)``.
    max_boxes : int, default 64
        Maximum number of box slots retained per query.

    Returns
    -------
    MultiBox
        Pairwise intersection multi-box region.
    """
    r1 = _validate_multibox(R1, "R1")
    r2 = _validate_multibox(R2, "R2")
    max_b = _validate_int_ge("max_boxes", max_boxes, 1)
    r1, r2 = _align_multibox_batch(r1, r2)

    bsz, m1, dim = r1.cen.shape
    m2 = r2.cen.shape[1]
    total = m1 * m2
    if total == 0:
        return MultiBox(
            cen=th.empty((bsz, 0, dim), dtype=r1.cen.dtype, device=r1.cen.device),
            w=th.empty((bsz, 0, dim), dtype=r1.w.dtype, device=r1.w.device),
            active=th.empty((bsz, 0), dtype=th.bool, device=r1.active.device),
        )

    c1 = r1.cen.unsqueeze(2)
    w1 = r1.w.unsqueeze(2)
    c2 = r2.cen.unsqueeze(1)
    w2 = r2.w.unsqueeze(1)
    lo = th.maximum(c1 - w1, c2 - w2)
    hi = th.minimum(c1 + w1, c2 + w2)
    cen_ij = 0.5 * (lo + hi)
    w_ij = 0.5 * (hi - lo)

    non_empty = th.amin(w_ij, dim=-1) > 0.0
    act_ij = r1.active.unsqueeze(2) & r2.active.unsqueeze(1) & non_empty

    cen_flat = cen_ij.reshape(bsz, total, dim)
    w_flat = th.clamp_min(w_ij.reshape(bsz, total, dim), 0.0)
    act_flat = act_ij.reshape(bsz, total)

    if total <= max_b:
        return _compact_active_first(cen_flat, w_flat, act_flat)
    return _cap_boxes(MultiBox(cen=cen_flat, w=w_flat, active=act_flat), max_b)


def union(
    R1: MultiBox,
    R2: MultiBox,
    max_boxes: int = 64,
) -> MultiBox:
    """Compute the union of ``R1`` and ``R2`` by concatenating box slots and capping at ``max_boxes``.

    Guarantees:
    - Returns a :class:`MultiBox` of shape ``(B, min(M1 + M2, max_boxes), d)``.
    - When ``M1 + M2 <= max_boxes``, preserves the exact concatenated slot order
      ``[R1, R2]``.
    - When ``M1 + M2 > max_boxes``, retains the ``max_boxes`` largest-volume
      active boxes.

    Non-guarantees:
    - Does not merge overlapping boxes; when ``M1 + M2 > max_boxes``, smaller
      boxes are dropped.

    Parameters
    ----------
    R1 : MultiBox
        First multi-box region of shape ``(B, M1, d)``.
    R2 : MultiBox
        Second multi-box region of shape ``(B, M2, d)``.
    max_boxes : int, default 64
        Maximum number of box slots retained per query.

    Returns
    -------
    MultiBox
        Union multi-box region.
    """
    r1 = _validate_multibox(R1, "R1")
    r2 = _validate_multibox(R2, "R2")
    max_b = _validate_int_ge("max_boxes", max_boxes, 1)
    r1, r2 = _align_multibox_batch(r1, r2)

    cen = th.cat([r1.cen, r2.cen], dim=1)
    w = th.cat([r1.w, r2.w], dim=1)
    act = th.cat([r1.active, r2.active], dim=1)
    act_mask = act.unsqueeze(-1)
    cen = th.where(act_mask, cen, th.zeros_like(cen))
    w = th.where(act_mask, th.clamp_min(w, 0.0), th.zeros_like(w))
    out = MultiBox(cen=cen, w=w, active=act)
    if cen.shape[1] > max_b:
        return _cap_boxes(out, max_b)
    return out


def class_boxes(
    module: GroundedELModule,
    class_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]] = None,
) -> MultiBox:
    """Materialise the :math:`K` class boxes of ``class_ids`` as a :class:`MultiBox`.

    Guarantees:
    - Returns a :class:`MultiBox` of shape ``(B, n_components, d)`` matching
      :meth:`GroundedELModule.class_regions`.
    - When ``module.bot_id`` is set, rows matching ``bot_id`` have all slots
      inactive with ``w == 0.0``.
    - When ``module.top_id`` is set, rows matching ``top_id`` have slot ``0``
      active with half-width ``1e6`` and remaining slots inactive.

    Non-guarantees:
    - Does not cap ``n_components``.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module holding class box parameters.
    class_ids : Optional[Union[int, Sequence[int], numpy.ndarray, torch.Tensor]], default None
        1-D class indices in ``[0, module.nb_classes)``, or ``None`` for all classes.

    Returns
    -------
    MultiBox
        Class multi-box regions of shape ``(B, n_components, d)``.
    """
    _validate_module(module, require_translation=False)
    device = module.class_centers.device
    if class_ids is None:
        idx = th.arange(module.nb_classes, dtype=th.long, device=device)
    else:
        idx = _to_1d_long(class_ids, device, "class_ids")

    reg = module.class_regions(idx)
    cen = reg.cen
    w = reg.w
    act = reg.active

    if module.bot_id is not None and idx.numel() > 0:
        is_bot = (idx == module.bot_id).unsqueeze(1)  # (B, 1)
        act = act & (~is_bot)
        w = th.where(act.unsqueeze(-1), w, th.zeros_like(w))

    if module.top_id is not None and idx.numel() > 0:
        is_top = idx == module.top_id  # (B,)
        is_top_2d = is_top.unsqueeze(1)  # (B, 1)
        is_top_3d = is_top.view(-1, 1, 1)  # (B, 1, 1)
        top_cen = th.zeros_like(cen)
        top_w = th.zeros_like(w)
        top_w[:, 0, :] = _TOP_DEPTH_CEIL
        top_act = th.zeros_like(act)
        top_act[:, 0] = True
        cen = th.where(is_top_3d, top_cen, cen)
        w = th.where(is_top_3d, top_w, w)
        act = th.where(is_top_2d, top_act, act)

    return MultiBox(cen=cen, w=w, active=act)


def depth(
    points: th.Tensor,
    R: MultiBox,
    crisp: bool = True,
) -> th.Tensor:
    """Compute signed interior depth :math:`\\max_m \\min_j (w_{m,j} - |p_{n,j} - c_{m,j}|)` of ``points`` in ``R``.

    Guarantees:
    - Returns a 2-D tensor of shape ``(B, N)`` where ``B`` is the batch size of
      ``R`` and ``N`` is the number of candidate points.
    - Under ``crisp=True``, entry ``(b, n)`` is strictly positive if and only if
      point ``n`` lies strictly inside the open interior of at least one active
      box of ``R[b]``.
    - Query rows in ``R`` with zero active boxes return the finite floor ``-1e6``.

    Non-guarantees:
    - Under ``crisp=True``, gradients flow only through the winning active box
      and bottleneck coordinate.

    Parameters
    ----------
    points : torch.Tensor
        Candidate point coordinates of shape ``(N, d)`` or ``(B, N, d)`` (or 1-D ``(d,)``).
    R : MultiBox
        Query multi-box regions of shape ``(B, M, d)``.
    crisp : bool, default True
        If ``True``, uses hard ``max`` over active boxes; if ``False``, uses
        smooth ``logsumexp`` with temperature ``0.1``.

    Returns
    -------
    torch.Tensor
        Signed depths of shape ``(B, N)``.
    """
    R_val = _validate_multibox(R, "R")
    if not isinstance(points, th.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
    if isinstance(crisp, bool) is False:
        raise TypeError(f"crisp must be a bool, got {type(crisp).__name__}")

    bsz, m_slots, dim = R_val.cen.shape
    if points.ndim == 1:
        if points.shape[0] != dim:
            raise ValueError(
                f"1-D points must have length d={dim}, got {tuple(points.shape)}"
            )
        pts = points.unsqueeze(0)  # (1, d)
        p_exp = pts.unsqueeze(0).unsqueeze(2)  # (1, 1, 1, d)
        n_pts = 1
    elif points.ndim == 2:
        if points.shape[1] != dim:
            raise ValueError(
                f"2-D points must have shape (N, {dim}), got {tuple(points.shape)}"
            )
        n_pts = points.shape[0]
        p_exp = points.unsqueeze(0).unsqueeze(2)  # (1, N, 1, d)
    elif points.ndim == 3:
        if points.shape[0] != bsz or points.shape[2] != dim:
            raise ValueError(
                f"3-D points must have shape ({bsz}, N, {dim}), got {tuple(points.shape)}"
            )
        n_pts = points.shape[1]
        p_exp = points.unsqueeze(2)  # (B, N, 1, d)
    else:
        raise ValueError(
            f"points must have 1, 2, or 3 dimensions, got shape {tuple(points.shape)}"
        )

    if bsz == 0 or n_pts == 0:
        return th.empty((bsz, n_pts), dtype=R_val.cen.dtype, device=R_val.cen.device)
    if m_slots == 0:
        return th.full(
            (bsz, n_pts),
            _EMPTY_DEPTH_FLOOR,
            dtype=R_val.cen.dtype,
            device=R_val.cen.device,
        )

    p_exp = p_exp.to(dtype=R_val.cen.dtype, device=R_val.cen.device)
    coord_depth = R_val.w.unsqueeze(1) - th.abs(p_exp - R_val.cen.unsqueeze(1))  # (B, N, M, d)
    slot_depth = th.amin(coord_depth, dim=-1)  # (B, N, M)

    act_mask = R_val.active.unsqueeze(1)  # (B, 1, M)
    has_active = act_mask.any(dim=-1, keepdim=True)  # (B, 1, 1)
    inactive_fill = th.where(
        has_active,
        th.full_like(slot_depth, float("-inf")),
        th.full_like(slot_depth, _EMPTY_DEPTH_FLOOR),
    )
    masked = th.where(act_mask, slot_depth, inactive_fill)
    if crisp:
        return th.amax(masked, dim=-1)
    lse = _LSE_DEFAULT_TAU * th.logsumexp(masked / _LSE_DEFAULT_TAU, dim=-1)
    return th.where(
        has_active.squeeze(-1),
        lse,
        th.full_like(lse, _EMPTY_DEPTH_FLOOR),
    )


def overlap_depth(
    R: MultiBox,
    Q: MultiBox,
) -> th.Tensor:
    """Compute pairwise box overlap depth :math:`\\max_{m,q} \\min_j (w_{m,j} + w_{q,j} - |c_{m,j} - c_{q,j}|)`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Entry ``b`` is strictly positive if and only if there is at least one
      active box in ``R[b]`` and at least one active box in ``Q[b]`` whose open
      interiors intersect.
    - Rows where either ``R[b]`` or ``Q[b]`` has zero active boxes return the
      finite floor ``-1e6``.

    Non-guarantees:
    - Gradients flow only through the winning active pair ``(m, q)`` and
      bottleneck coordinate.

    Parameters
    ----------
    R : MultiBox
        First multi-box region of shape ``(B, M, d)`` (or ``(1, M, d)``).
    Q : MultiBox
        Second multi-box region of shape ``(B, K, d)`` (or ``(1, K, d)``).

    Returns
    -------
    torch.Tensor
        Signed overlap depths of shape ``(B,)``.
    """
    r_val = _validate_multibox(R, "R")
    q_val = _validate_multibox(Q, "Q")
    r_val, q_val = _align_multibox_batch(r_val, q_val)

    bsz, m_slots, dim = r_val.cen.shape
    k_slots = q_val.cen.shape[1]
    total = m_slots * k_slots
    if bsz == 0:
        return th.empty((0,), dtype=r_val.cen.dtype, device=r_val.cen.device)
    if total == 0:
        return th.full(
            (bsz,), _EMPTY_DEPTH_FLOOR, dtype=r_val.cen.dtype, device=r_val.cen.device
        )

    q_cen = q_val.cen.to(dtype=r_val.cen.dtype, device=r_val.cen.device)
    q_w = q_val.w.to(dtype=r_val.w.dtype, device=r_val.w.device)
    q_act = q_val.active.to(device=r_val.active.device)

    coord_ov = (
        r_val.w.unsqueeze(2)
        + q_w.unsqueeze(1)
        - th.abs(r_val.cen.unsqueeze(2) - q_cen.unsqueeze(1))
    )  # (B, M, K, d)
    pair_ov = th.amin(coord_ov, dim=-1).reshape(bsz, total)
    pair_act = (r_val.active.unsqueeze(2) & q_act.unsqueeze(1)).reshape(bsz, total)

    has_active = pair_act.any(dim=1, keepdim=True)
    inactive_fill = th.where(
        has_active,
        th.full_like(pair_ov, float("-inf")),
        th.full_like(pair_ov, _EMPTY_DEPTH_FLOOR),
    )
    masked = th.where(pair_act, pair_ov, inactive_fill)
    return th.amax(masked, dim=1)


def score_entities(
    module: GroundedELModule,
    R: MultiBox,
    candidate_element_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]] = None,
    chunk: int = 65536,
) -> th.Tensor:
    """Score candidate domain elements against query region ``R`` by crisp depth.

    Guarantees:
    - Returns a 2-D tensor of shape ``(B, N)`` where ``N`` is the number of
      candidate elements (``module.nb_elements`` when ``candidate_element_ids is None``).
    - Entry ``(b, n)`` is strictly positive if and only if candidate ``n`` lies
      strictly inside at least one active box of ``R[b]``.
    - Evaluates candidates in chunks of at most ``chunk`` elements.

    Non-guarantees:
    - Does not filter easy answers; filtering is applied by the caller.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module holding element positions.
    R : MultiBox
        Compiled query multi-box region of shape ``(B, M, d)``.
    candidate_element_ids : Optional[Union[int, Sequence[int], numpy.ndarray, torch.Tensor]], default None
        1-D candidate element indices in ``[0, module.nb_elements)``, or ``None``
        for all elements ``0 .. module.nb_elements - 1``.
    chunk : int, default 65536
        Candidate chunk size :math:`\\ge 1`.

    Returns
    -------
    torch.Tensor
        Crisp depth scores of shape ``(B, N)``.
    """
    _validate_module(module, require_translation=False)
    R_val = _validate_multibox(R, "R", expected_dim=module.embed_dim)
    chunk_sz = _validate_int_ge("chunk", chunk, 1)
    device = module.elem_pos.weight.device

    if candidate_element_ids is None:
        cand_idx = th.arange(module.nb_elements, dtype=th.long, device=device)
    else:
        cand_idx = _to_1d_long(candidate_element_ids, device, "candidate_element_ids")

    n_cand = cand_idx.shape[0]
    if n_cand <= chunk_sz:
        pts = module.positions(cand_idx)
        return depth(pts, R_val, crisp=True)

    chunks: List[th.Tensor] = []
    for start in range(0, n_cand, chunk_sz):
        pts_chunk = module.positions(cand_idx[start : start + chunk_sz])
        chunks.append(depth(pts_chunk, R_val, crisp=True))
    return th.cat(chunks, dim=1)


def score_concepts(
    module: GroundedELModule,
    R: MultiBox,
    class_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]] = None,
) -> th.Tensor:
    """Score candidate ontology classes against query region ``R`` by box overlap depth.

    Evaluates :math:`\\max_{m, k} \\min_j (w_{m,j} + w_{C,k,j} - |c_{m,j} - c_{C,k,j}|)`
    between every query in ``R`` and every candidate class :math:`C`.

    Guarantees:
    - Returns a 2-D tensor of shape ``(B, C)`` where ``C`` is the number of
      candidate classes (``module.nb_classes`` when ``class_ids is None``).
    - Entry ``(b, c)`` is strictly positive if and only if at least one active
      box of ``R[b]`` overlaps at least one active box of class ``c``.

    Non-guarantees:
    - Gradients flow only through the winning box pair and bottleneck coordinate.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module holding class box parameters.
    R : MultiBox
        Compiled query multi-box region of shape ``(B, M, d)``.
    class_ids : Optional[Union[int, Sequence[int], numpy.ndarray, torch.Tensor]], default None
        1-D class indices in ``[0, module.nb_classes)``, or ``None`` for all
        classes ``0 .. module.nb_classes - 1``.

    Returns
    -------
    torch.Tensor
        Concept overlap depth scores of shape ``(B, C)``.
    """
    _validate_module(module, require_translation=False)
    R_val = _validate_multibox(R, "R", expected_dim=module.embed_dim)
    c_boxes = class_boxes(module, class_ids)

    bsz, m_slots, dim = R_val.cen.shape
    n_cls, k_slots, _ = c_boxes.cen.shape
    total_pairs = m_slots * k_slots
    if bsz == 0 or n_cls == 0:
        return th.empty((bsz, n_cls), dtype=R_val.cen.dtype, device=R_val.cen.device)
    if total_pairs == 0:
        return th.full(
            (bsz, n_cls),
            _EMPTY_DEPTH_FLOOR,
            dtype=R_val.cen.dtype,
            device=R_val.cen.device,
        )

    c_cen = c_boxes.cen.to(dtype=R_val.cen.dtype, device=R_val.cen.device)
    c_w = c_boxes.w.to(dtype=R_val.w.dtype, device=R_val.w.device)
    c_act = c_boxes.active.to(device=R_val.active.device)

    coord_ov = (
        R_val.w.unsqueeze(1).unsqueeze(3)
        + c_w.unsqueeze(0).unsqueeze(2)
        - th.abs(R_val.cen.unsqueeze(1).unsqueeze(3) - c_cen.unsqueeze(0).unsqueeze(2))
    )  # (B, C, M, K, d)
    pair_ov = th.amin(coord_ov, dim=-1).reshape(bsz, n_cls, total_pairs)
    pair_act = (
        R_val.active.unsqueeze(1).unsqueeze(3) & c_act.unsqueeze(0).unsqueeze(2)
    ).reshape(bsz, n_cls, total_pairs)

    has_active = pair_act.any(dim=-1, keepdim=True)  # (B, C, 1)
    inactive_fill = th.where(
        has_active,
        th.full_like(pair_ov, float("-inf")),
        th.full_like(pair_ov, _EMPTY_DEPTH_FLOOR),
    )
    masked = th.where(pair_act, pair_ov, inactive_fill)
    return th.amax(masked, dim=-1)


def project_exact(
    module: GroundedELModule,
    R: MultiBox,
    rel_ids: Union[int, Sequence[int], np.ndarray, th.Tensor],
    element_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]] = None,
    max_witnesses: int = 256,
    max_boxes: int = 64,
) -> MultiBox:
    """Project region ``R`` along ``rel_ids`` using domain elements with ``depth(p_z, R) > 0`` as intermediate witnesses.

    For each query in ``R``, identifies elements :math:`z` in ``element_ids``
    satisfying :math:`\\operatorname{depth}(p_z, R) > 0` (retaining at most the
    top ``max_witnesses`` elements by depth), then forms
    :math:`\\bigcup_{z, k} (p_z + c_{r,k},\\, w_{r,k})` and caps at ``max_boxes``.

    Guarantees:
    - Returns a :class:`MultiBox` of shape ``(B, M_out, d)`` with ``M_out <= max_boxes``.
    - When the number of witness elements is at most ``max_witnesses`` and
      ``num_witnesses * K_r <= max_boxes``, ``score_entities(module, project_exact(...)) > 0``
      is true for element :math:`y` if and only if there exists a witness element
      :math:`z` in ``element_ids`` with :math:`\\operatorname{depth}(p_z, R) > 0`
      and :math:`\\operatorname{role\\_depth}(p_z, p_y, r, \\text{crisp}=\\text{True}) > 0`.
    - Raises :class:`ValueError` if ``module.role_mode != "translation"``.

    Non-guarantees:
    - Witness selection uses a hard threshold ``depth(p_z, R) > 0``; gradients
      with respect to ``R`` do not propagate through the discrete witness filter.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module in ``"translation"`` role mode.
    R : MultiBox
        Input multi-box region of shape ``(B, M, d)``.
    rel_ids : Union[int, Sequence[int], numpy.ndarray, torch.Tensor]
        Role indices of shape ``(B,)`` (or scalar broadcast to ``B``) in
        ``[0, module.nb_rels)``.
    element_ids : Optional[Union[int, Sequence[int], numpy.ndarray, th.Tensor]], default None
        Candidate witness element indices in ``[0, module.nb_elements)``, or
        ``None`` for all elements ``0 .. module.nb_elements - 1``.
    max_witnesses : int, default 256
        Maximum number of witness elements retained per query :math:`\\ge 1`.
    max_boxes : int, default 64
        Maximum number of box slots retained per query :math:`\\ge 1`.

    Returns
    -------
    MultiBox
        Witness-projected multi-box region.
    """
    _validate_module(module, require_translation=True)
    R_val = _validate_multibox(R, "R", expected_dim=module.embed_dim)
    max_w = _validate_int_ge("max_witnesses", max_witnesses, 1)
    max_b = _validate_int_ge("max_boxes", max_boxes, 1)

    bsz, _, dim = R_val.cen.shape
    device = R_val.cen.device
    r_idx = _to_1d_long(rel_ids, device, "rel_ids")
    if r_idx.shape[0] == 1 and bsz > 1:
        r_idx = r_idx.expand(bsz)
    elif bsz == 0 and r_idx.shape[0] == 0:
        pass
    elif r_idx.shape[0] != bsz:
        raise ValueError(
            f"Incompatible batch sizes between R ({bsz}) and rel_ids ({r_idx.shape[0]})"
        )

    if element_ids is None:
        elem_idx = th.arange(module.nb_elements, dtype=th.long, device=device)
    else:
        elem_idx = _to_1d_long(element_ids, device, "element_ids")

    n_elem = elem_idx.shape[0]
    rel_reg = module.translation_regions(r_idx)
    kr = rel_reg.cen.shape[1]
    if bsz == 0 or n_elem == 0 or kr == 0:
        return MultiBox(
            cen=th.empty((bsz, 0, dim), dtype=R_val.cen.dtype, device=device),
            w=th.empty((bsz, 0, dim), dtype=R_val.w.dtype, device=device),
            active=th.empty((bsz, 0), dtype=th.bool, device=device),
        )

    pts = module.positions(elem_idx)  # (N, d)
    dep = depth(pts, R_val, crisp=True)  # (B, N)
    valid_w = dep > 0.0  # (B, N)
    masked_dep = th.where(valid_w, dep, th.full_like(dep, float("-inf")))

    w_keep = min(n_elem, max_w)
    _, top_w_idx = th.topk(masked_dep, k=w_keep, dim=1, largest=True, sorted=True)  # (B, W)
    w_active = th.gather(valid_w, 1, top_w_idx)  # (B, W)
    w_pos = pts[top_w_idx].to(dtype=R_val.cen.dtype)  # (B, W, d)

    r_cen = rel_reg.cen.to(dtype=R_val.cen.dtype)
    r_w = rel_reg.w.to(dtype=R_val.w.dtype)
    total = w_keep * kr
    cen_wk = (w_pos.unsqueeze(2) + r_cen.unsqueeze(1)).reshape(bsz, total, dim)
    w_wk = r_w.unsqueeze(1).expand(bsz, w_keep, kr, dim).reshape(bsz, total, dim)
    act_wk = (w_active.unsqueeze(2) & rel_reg.active.unsqueeze(1)).reshape(bsz, total)

    act_mask = act_wk.unsqueeze(-1)
    cen_wk = th.where(act_mask, cen_wk, th.zeros_like(cen_wk))
    w_wk = th.where(act_mask, w_wk, th.zeros_like(w_wk))
    out = MultiBox(cen=cen_wk, w=w_wk, active=act_wk)
    if total > max_b:
        return _cap_boxes(out, max_b)
    return _compact_active_first(out.cen, out.w, out.active)


def _project_type_rel(
    module: GroundedELModule,
    R: MultiBox,
    max_boxes: int,
) -> MultiBox:
    """Execute the ``type_rel`` concept-level step: union of ``class_boxes(C)`` for classes overlapping ``R``."""
    bsz, _, dim = R.cen.shape
    all_cls = class_boxes(module, None)  # (C, K, d)
    n_cls, k_slots, _ = all_cls.cen.shape
    total = n_cls * k_slots
    if total == 0:
        return MultiBox(
            cen=th.empty((bsz, 0, dim), dtype=R.cen.dtype, device=R.cen.device),
            w=th.empty((bsz, 0, dim), dtype=R.w.dtype, device=R.w.device),
            active=th.empty((bsz, 0), dtype=th.bool, device=R.active.device),
        )

    ov = score_concepts(module, R)  # (B, C)
    cls_match = ov > 0.0  # (B, C)

    cen_all = (
        all_cls.cen.to(dtype=R.cen.dtype)
        .reshape(1, total, dim)
        .expand(bsz, total, dim)
    )
    w_all = (
        all_cls.w.to(dtype=R.w.dtype)
        .reshape(1, total, dim)
        .expand(bsz, total, dim)
    )
    act_all = (cls_match.unsqueeze(2) & all_cls.active.unsqueeze(0)).reshape(bsz, total)

    if total <= max_boxes:
        return _compact_active_first(cen_all, w_all, act_all)
    return _cap_boxes(MultiBox(cen=cen_all, w=w_all, active=act_all), max_boxes)


def _merge_multibox_rows(
    mask: th.Tensor,
    R_true: MultiBox,
    R_false: MultiBox,
) -> MultiBox:
    """Select rows from ``R_true`` where ``mask`` is ``True`` and ``R_false`` otherwise, padding slot counts."""
    bsz, m_t, dim = R_true.cen.shape
    m_f = R_false.cen.shape[1]
    m_max = max(m_t, m_f)

    def _pad(mb: MultiBox, target_m: int) -> MultiBox:
        cur_m = mb.cen.shape[1]
        if cur_m == target_m:
            return mb
        pad_n = target_m - cur_m
        pad_cen = th.zeros((bsz, pad_n, dim), dtype=mb.cen.dtype, device=mb.cen.device)
        pad_w = th.zeros((bsz, pad_n, dim), dtype=mb.w.dtype, device=mb.w.device)
        pad_act = th.zeros((bsz, pad_n), dtype=th.bool, device=mb.active.device)
        return MultiBox(
            cen=th.cat([mb.cen, pad_cen], dim=1),
            w=th.cat([mb.w, pad_w], dim=1),
            active=th.cat([mb.active, pad_act], dim=1),
        )

    rt = _pad(R_true, m_max)
    rf = _pad(R_false, m_max)
    mask_2d = mask.unsqueeze(1)
    mask_3d = mask.view(bsz, 1, 1)
    return MultiBox(
        cen=th.where(mask_3d, rt.cen, rf.cen),
        w=th.where(mask_3d, rt.w, rf.w),
        active=th.where(mask_2d, rt.active, rf.active),
    )


def _is_rel_tuple(s: Any) -> bool:
    """Return True if ``s`` is a non-empty tuple of ``'r'`` tokens."""
    return isinstance(s, tuple) and len(s) > 0 and all(x == "r" for x in s)


def _apply_hop(
    module: GroundedELModule,
    R: MultiBox,
    hop_rels_tuple: Tuple[int, ...],
    *,
    is_first_hop: bool,
    anchor_cls_ids: Optional[th.Tensor],
    all_anchors_are_classes: bool,
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    max_boxes: int,
    exact: bool,
    element_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]],
    max_witnesses: int,
) -> MultiBox:
    """Apply a single relation hop across the query batch without Python loops over the batch."""
    device = module.elem_pos.weight.device
    rel_set = set(hop_rels_tuple)
    r_t = _to_1d_long(hop_rels_tuple, device, "rel_ids")

    has_type_inv = (
        is_first_hop
        and type_inv_rel is not None
        and anchor_cls_ids is not None
        and (type_inv_rel in rel_set)
    )
    only_type_inv = (
        has_type_inv
        and rel_set == {type_inv_rel}
        and all_anchors_are_classes
    )
    has_type = type_rel is not None and (type_rel in rel_set)
    only_type = has_type and (not has_type_inv) and rel_set == {type_rel}
    only_normal = (not has_type_inv) and (not has_type)

    if only_type_inv:
        assert anchor_cls_ids is not None
        out_inv = class_boxes(module, anchor_cls_ids)
        return _cap_boxes(out_inv, max_boxes) if out_inv.cen.shape[1] > max_boxes else out_inv

    if only_type:
        return _project_type_rel(module, R, max_boxes)

    if only_normal:
        if exact and not is_first_hop:
            return project_exact(
                module,
                R,
                r_t,
                element_ids=element_ids,
                max_witnesses=max_witnesses,
                max_boxes=max_boxes,
            )
        return project(module, R, r_t, max_boxes=max_boxes)

    # Mixed batch: compute active branch types and merge by row mask
    special_rels = set()
    if has_type_inv and all_anchors_are_classes and type_inv_rel is not None:
        special_rels.add(type_inv_rel)
    if has_type and type_rel is not None:
        special_rels.add(type_rel)
    has_normal = len(rel_set - special_rels) > 0

    cur: Optional[MultiBox] = None
    if has_normal:
        safe_r = r_t
        if type_inv_rel is not None:
            safe_r = th.where(safe_r == type_inv_rel, th.zeros_like(safe_r), safe_r)
        if type_rel is not None:
            safe_r = th.where(safe_r == type_rel, th.zeros_like(safe_r), safe_r)
        if exact and not is_first_hop:
            cur = project_exact(
                module,
                R,
                safe_r,
                element_ids=element_ids,
                max_witnesses=max_witnesses,
                max_boxes=max_boxes,
            )
        else:
            cur = project(module, R, safe_r, max_boxes=max_boxes)

    if has_type:
        r_type = _project_type_rel(module, R, max_boxes)
        if cur is None:
            cur = r_type
        else:
            t_mask = r_t == type_rel
            cur = _merge_multibox_rows(t_mask, r_type, cur)

    if has_type_inv:
        assert anchor_cls_ids is not None
        inv_mask = (r_t == type_inv_rel) & (anchor_cls_ids >= 0)
        safe_cls = th.clamp_min(anchor_cls_ids, 0)
        r_inv = class_boxes(module, safe_cls)
        if r_inv.cen.shape[1] > max_boxes:
            r_inv = _cap_boxes(r_inv, max_boxes)
        if cur is None:
            cur = r_inv
        else:
            cur = _merge_multibox_rows(inv_mask, r_inv, cur)

    assert cur is not None
    return _cap_boxes(cur, max_boxes) if cur.cen.shape[1] > max_boxes else cur


def _compile_recursive(
    module: GroundedELModule,
    structure: Any,
    queries: Sequence[Any],
    *,
    e2el_t: Optional[th.Tensor],
    c2e_t: Optional[th.Tensor],
    c2e_arr: Optional[np.ndarray],
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    max_boxes: int,
    exact: bool,
    element_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]],
    max_witnesses: int,
) -> MultiBox:
    """Recursively compile ``queries`` matching ``structure``."""
    bsz = len(queries)
    device = module.elem_pos.weight.device

    # Case 1: Path starting from an anchor ('e', ('r', ...)) or ('e',) or 'e'
    if structure == "e" or (
        isinstance(structure, tuple) and len(structure) >= 1 and structure[0] == "e"
    ):
        if structure == "e":
            ents_seq: Any = queries
            hop_tuples: Tuple[Tuple[int, ...], ...] = ()
            n_hops = 0
        elif len(structure) == 1:
            (ents_seq,) = tuple(zip(*queries))
            hop_tuples = ()
            n_hops = 0
        elif (
            len(structure) == 2
            and isinstance(structure[1], tuple)
            and all(x == "r" for x in structure[1])
        ):
            n_hops = len(structure[1])
            unpacked = tuple(zip(*queries))
            if len(unpacked) != 2:
                raise ValueError(
                    f"Queries do not match anchor structure {structure!r}"
                )
            ents_seq, rels_seq = unpacked
            if n_hops > 0:
                hop_tuples = tuple(zip(*rels_seq))
                if len(hop_tuples) != n_hops:
                    raise ValueError(
                        f"Query relation tuples do not match structure {structure!r}"
                    )
            else:
                hop_tuples = ()
        else:
            raise ValueError(f"Unsupported anchor query structure: {structure!r}")

        ent_ids = _to_1d_long(ents_seq, device, "anchor_ids")
        ent_arr = np.asarray(ents_seq, dtype=np.int64)

        if c2e_t is not None and c2e_arr is not None:
            if ent_arr.min() < 0 or ent_arr.max() >= c2e_arr.shape[0]:
                raise ValueError(
                    f"Anchor entity id out of range [0, {c2e_arr.shape[0]}) for class_of_entity"
                )
            anchor_cls_ids: Optional[th.Tensor] = c2e_t[ent_ids]
            all_anchors_are_classes = bool(np.all(c2e_arr[ent_arr] >= 0))
        elif type_inv_rel is not None:
            if e2el_t is not None:
                if ent_arr.min() < 0 or ent_arr.max() >= int(e2el_t.shape[0]):
                    raise ValueError(
                        f"Anchor entity id out of range [0, {int(e2el_t.shape[0])}) for entity_to_element"
                    )
                mapped_ids = e2el_t[ent_ids]
            else:
                mapped_ids = ent_ids
            is_cls_elem = (mapped_ids >= 0) & (mapped_ids < module.nb_classes)
            anchor_cls_ids = th.where(
                is_cls_elem, mapped_ids, th.full_like(mapped_ids, -1)
            )
            all_anchors_are_classes = (
                e2el_t is None
                and bool(np.all((ent_arr >= 0) & (ent_arr < module.nb_classes)))
            )
        else:
            anchor_cls_ids = None
            all_anchors_are_classes = False

        if e2el_t is not None:
            if ent_arr.min() < 0 or ent_arr.max() >= int(e2el_t.shape[0]):
                raise ValueError(
                    f"Anchor entity id out of range [0, {int(e2el_t.shape[0])}) for entity_to_element"
                )
            elem_ids = e2el_t[ent_ids]
        else:
            elem_ids = ent_ids

        hop0_only_type_inv = (
            n_hops > 0
            and type_inv_rel is not None
            and set(hop_tuples[0]) == {type_inv_rel}
            and all_anchors_are_classes
        )
        if hop0_only_type_inv:
            R = MultiBox(
                cen=th.zeros(
                    (bsz, 1, module.embed_dim),
                    dtype=module.elem_pos.weight.dtype,
                    device=device,
                ),
                w=th.zeros(
                    (bsz, 1, module.embed_dim),
                    dtype=module.elem_pos.weight.dtype,
                    device=device,
                ),
                active=th.ones((bsz, 1), dtype=th.bool, device=device),
            )
        else:
            if (
                n_hops > 0
                and type_inv_rel is not None
                and (type_inv_rel in set(hop_tuples[0]))
                and anchor_cls_ids is not None
            ):
                r0_t = _to_1d_long(hop_tuples[0], device, "rel_ids")
                is_replaced = (r0_t == type_inv_rel) & (anchor_cls_ids >= 0)
                safe_elem_ids = th.where(
                    is_replaced & (elem_ids < 0),
                    th.zeros_like(elem_ids),
                    elem_ids,
                )
            else:
                safe_elem_ids = elem_ids
            R = anchor(module, safe_elem_ids)

        for hop in range(n_hops):
            R = _apply_hop(
                module,
                R,
                hop_tuples[hop],
                is_first_hop=(hop == 0),
                anchor_cls_ids=anchor_cls_ids if hop == 0 else None,
                all_anchors_are_classes=all_anchors_are_classes if hop == 0 else False,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                max_boxes=max_boxes,
                exact=exact,
                element_ids=element_ids,
                max_witnesses=max_witnesses,
            )
        return R

    if not isinstance(structure, tuple) or len(structure) < 2:
        raise ValueError(f"Unsupported query structure: {structure!r}")

    cols = tuple(zip(*queries))

    # Case 2: Compound projection (sub_structure, ('r', ...)) such as ip or up
    if len(structure) == 2 and _is_rel_tuple(structure[1]):
        if len(cols) != 2:
            raise ValueError(f"Queries do not match compound structure {structure!r}")
        sub_struct = structure[0]
        rel_spec = structure[1]
        n_hops = len(rel_spec)
        sub_queries, rels_seq = cols
        hop_tuples = tuple(zip(*rels_seq))
        if len(hop_tuples) != n_hops:
            raise ValueError(
                f"Query relation tuples do not match structure {structure!r}"
            )
        R = _compile_recursive(
            module,
            sub_struct,
            sub_queries,
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            max_boxes=max_boxes,
            exact=exact,
            element_ids=element_ids,
            max_witnesses=max_witnesses,
        )
        for hop in range(n_hops):
            R = _apply_hop(
                module,
                R,
                hop_tuples[hop],
                is_first_hop=False,
                anchor_cls_ids=None,
                all_anchors_are_classes=False,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                max_boxes=max_boxes,
                exact=exact,
                element_ids=element_ids,
                max_witnesses=max_witnesses,
            )
        return R

    # Case 3: Union (sub_1, sub_2, ..., ('u',))
    if structure[-1] == ("u",):
        if len(structure) < 3:
            raise ValueError(
                f"Union structure must have at least 2 disjuncts before ('u',), got {structure!r}"
            )
        n_disj = len(structure) - 1
        if len(cols) < n_disj:
            raise ValueError(f"Queries do not match union structure {structure!r}")
        R_acc = _compile_recursive(
            module,
            structure[0],
            cols[0],
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            max_boxes=max_boxes,
            exact=exact,
            element_ids=element_ids,
            max_witnesses=max_witnesses,
        )
        for i in range(1, n_disj):
            R_i = _compile_recursive(
                module,
                structure[i],
                cols[i],
                e2el_t=e2el_t,
                c2e_t=c2e_t,
                c2e_arr=c2e_arr,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                max_boxes=max_boxes,
                exact=exact,
                element_ids=element_ids,
                max_witnesses=max_witnesses,
            )
            R_acc = union(R_acc, R_i, max_boxes=max_boxes)
        return R_acc

    # Case 4: Intersection (sub_1, sub_2, ...)
    n_conj = len(structure)
    if len(cols) != n_conj:
        raise ValueError(f"Queries do not match intersection structure {structure!r}")
    R_acc = _compile_recursive(
        module,
        structure[0],
        cols[0],
        e2el_t=e2el_t,
        c2e_t=c2e_t,
        c2e_arr=c2e_arr,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        max_boxes=max_boxes,
        exact=exact,
        element_ids=element_ids,
        max_witnesses=max_witnesses,
    )
    for i in range(1, n_conj):
        R_i = _compile_recursive(
            module,
            structure[i],
            cols[i],
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            max_boxes=max_boxes,
            exact=exact,
            element_ids=element_ids,
            max_witnesses=max_witnesses,
        )
        R_acc = intersect(R_acc, R_i, max_boxes=max_boxes)
    return R_acc


def compile_query(
    module: GroundedELModule,
    structure: Any,
    queries: Sequence[Any],
    *,
    type_rel: Optional[int] = None,
    type_inv_rel: Optional[int] = None,
    class_of_entity: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    entity_to_element: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    max_boxes: int = 64,
    exact: bool = False,
    element_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]] = None,
    max_witnesses: int = 256,
) -> MultiBox:
    """Compile a batch of Query2Box-structured ``queries`` into a :class:`MultiBox`.

    Supports standard Query2Box / BetaE structures including ``('e', ('r',))``,
    ``('e', ('r', 'r'))``, ``('e', ('r', 'r', 'r'))``,
    ``(('e', ('r',)), ('e', ('r',)))``,
    ``(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))``,
    ``(('e', ('r',)), ('e', ('r',)), ('u',))``,
    ``((('e', ('r',)), ('e', ('r',))), ('r',))``,
    ``(('e', ('r', 'r')), ('e', ('r',)))``, and
    ``((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))``.

    Special relations:
    - If ``rel == type_inv_rel`` at the first hop from an anchor of a class
      entity :math:`A`, the region becomes :func:`class_boxes` of :math:`A`.
    - If ``rel == type_rel``, the region becomes the union of
      :func:`class_boxes` of all classes :math:`C` with
      ``overlap_depth(region, class_boxes(C)) > 0``, capped at ``max_boxes``.
    - If ``exact=True``, projections after the first hop use
      :func:`project_exact` over ``element_ids``.

    Guarantees:
    - Returns a single :class:`MultiBox` of shape ``(B, M, d)`` with
      ``M <= max_boxes``; unions ``('u',)`` are represented directly as
      multi-boxes without exposing per-disjunct outputs to the caller.
    - When ``max_boxes`` is not exceeded at any step, the compiled region is
      exact with respect to Minkowski-sum and box-intersection semantics
      (or element-witness semantics when ``exact=True``).
    - Raises :class:`ValueError` if ``module.role_mode != "translation"``.

    Non-guarantees:
    - Approximates intermediate regions by their ``max_boxes`` largest-volume
      active boxes when the number of active boxes exceeds ``max_boxes``.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module in ``"translation"`` role mode.
    structure : Any
        Query2Box structure tuple describing the query syntax tree.
    queries : Sequence[Any]
        Batch of query instances matching ``structure``.
    type_rel : Optional[int], default None
        Optional relation id representing ``rdf:type`` (concept-level step).
    type_inv_rel : Optional[int], default None
        Optional relation id representing ``rdf:type^-`` (class-instance step).
    class_of_entity : Optional[Union[Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to class ids in
        ``[0, module.nb_classes)`` (or ``-1`` for non-class entities).
    entity_to_element : Optional[Union[Sequence[int], numpy.ndarray, th.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to module element ids in
        ``[0, module.nb_elements)``. When ``None``, anchor ids in ``queries``
        are used directly as module element ids.
    max_boxes : int, default 64
        Maximum number of box slots retained per query at each step.
    exact : bool, default False
        If ``True``, projections after the first hop use :func:`project_exact`.
    element_ids : Optional[Union[int, Sequence[int], numpy.ndarray, th.Tensor]], default None
        Optional candidate witness element ids forwarded to :func:`project_exact`
        when ``exact=True``.
    max_witnesses : int, default 256
        Maximum number of witness elements retained per query in
        :func:`project_exact` when ``exact=True``.

    Returns
    -------
    MultiBox
        Compiled multi-box region of shape ``(B, M, d)``.
    """
    _validate_module(module, require_translation=True)
    max_b = _validate_int_ge("max_boxes", max_boxes, 1)
    max_w = _validate_int_ge("max_witnesses", max_witnesses, 1)
    if not isinstance(exact, bool):
        raise TypeError(f"exact must be a bool, got {type(exact).__name__}")
    if not isinstance(queries, (list, tuple)):
        raise TypeError(
            f"queries must be a Sequence (list or tuple) of query instances, got {type(queries).__name__}"
        )

    device = module.elem_pos.weight.device
    if len(queries) == 0:
        return MultiBox(
            cen=th.empty(
                (0, 1, module.embed_dim),
                dtype=module.elem_pos.weight.dtype,
                device=device,
            ),
            w=th.empty(
                (0, 1, module.embed_dim),
                dtype=module.elem_pos.weight.dtype,
                device=device,
            ),
            active=th.empty((0, 1), dtype=th.bool, device=device),
        )

    e2el_t: Optional[th.Tensor] = None
    if entity_to_element is not None:
        e2el_t = _to_1d_long(entity_to_element, device, "entity_to_element")

    c2e_t: Optional[th.Tensor] = None
    c2e_arr: Optional[np.ndarray] = None
    if class_of_entity is not None:
        c2e_t = _to_1d_long(class_of_entity, device, "class_of_entity")
        c2e_arr = c2e_t.detach().cpu().numpy()

    return _compile_recursive(
        module,
        structure,
        queries,
        e2el_t=e2el_t,
        c2e_t=c2e_t,
        c2e_arr=c2e_arr,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        max_boxes=max_b,
        exact=exact,
        element_ids=element_ids,
        max_witnesses=max_w,
    )
