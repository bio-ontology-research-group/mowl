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
5. **Beam (max-min) query scorer and witness loss (:func:`score_query_beam`,
   :func:`score_query_witness`, :func:`query_witness_loss`):** Scores queries
   recursively over all domain elements using Gödel t-norm (max-min) combination
   of crisp depths over the top-``k`` witness elements per hop, and provides a
   differentiable target-restricted witness score and margin loss using soft
   depths at the module's current temperature ``module.tau``.

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
    "score_query_beam",
    "score_query_witness",
    "query_witness_loss",
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
    max_elems: int = 1 << 27,
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
        Candidate chunk size :math:`\\ge 1`; additionally capped so that the
        ``(B, chunk, M, d)`` depth intermediate holds at most ``max_elems`` scalars.
    max_elems : int, default 2**27
        Memory budget (number of scalars) for the per-chunk depth intermediate.

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
    # the depth intermediate is (B, n, M, d); bound it by ``max_elems`` scalars per chunk
    bsz, m_slots, dim = R_val.cen.shape
    per_cand = max(1, bsz * max(1, m_slots) * dim)
    eff_max_elems = _validate_int_ge("max_elems", max_elems, 1)
    if not th.is_grad_enabled() and device.type == "cpu":
        eff_max_elems = min(eff_max_elems, 1 << 21)
    chunk_sz = max(1, min(chunk_sz, eff_max_elems // per_cand))
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
    max_elems: int = 1 << 27,
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
    n_cls = c_boxes.cen.shape[0]
    if bsz == 0 or n_cls == 0:
        return th.empty((bsz, n_cls), dtype=R_val.cen.dtype, device=R_val.cen.device)
    # the overlap intermediate is (B, C, M, K, d); chunk over classes to bound it by ``max_elems`` scalars
    per_cls = max(1, bsz * max(1, m_slots * c_boxes.cen.shape[1]) * dim)
    c_chunk = max(1, _validate_int_ge("max_elems", max_elems, 1) // per_cls)
    if n_cls > c_chunk:
        parts = [
            _score_concepts_boxes(R_val, _slice_multibox(c_boxes, start, start + c_chunk))
            for start in range(0, n_cls, c_chunk)
        ]
        return th.cat(parts, dim=1)
    return _score_concepts_boxes(R_val, c_boxes)


def _slice_multibox(mb: MultiBox, start: int, stop: int) -> MultiBox:
    """Slice a :class:`MultiBox` along its first (batch or class) axis."""
    return MultiBox(cen=mb.cen[start:stop], w=mb.w[start:stop], active=mb.active[start:stop])


def _score_concepts_boxes(R_val: MultiBox, c_boxes: MultiBox) -> th.Tensor:
    """Overlap depth between query boxes ``R_val`` ``(B, M, d)`` and class boxes ``c_boxes`` ``(C, K, d)``."""
    bsz, m_slots, dim = R_val.cen.shape
    n_cls, k_slots, _ = c_boxes.cen.shape
    total_pairs = m_slots * k_slots
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
    dep = score_entities(module, R_val, candidate_element_ids=elem_idx)  # (B, N), memory-chunked crisp depth
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

    if (not is_first_hop) and type_inv_rel is not None and (type_inv_rel in rel_set):
        raise NotImplementedError(
            "compile_query does not support type_inv_rel at a non-first hop "
            "(class -> its instances after an intermediate step); use "
            "score_query_beam or score_query_witness for such queries"
        )

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
      ``type_inv_rel`` at a later hop raises :class:`NotImplementedError` (use
      :func:`score_query_beam` for such queries).
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


def _beam_project_rel(
    module: GroundedELModule,
    S: th.Tensor,
    rel_ids: th.Tensor,
    *,
    k: int,
    chunk: int,
    all_pos: th.Tensor,
    max_elems: int = 1 << 26,
) -> th.Tensor:
    """Project element score tensor ``S`` ``(B, E)`` along translation ``rel_ids`` using top-``k`` beam witnesses."""
    bsz, n_elem = S.shape
    dim = module.embed_dim
    device = S.device
    dtype = S.dtype

    rel_reg = module.translation_regions(rel_ids)
    kr = rel_reg.cen.shape[1]
    if bsz == 0 or n_elem == 0:
        return th.empty((bsz, n_elem), dtype=dtype, device=device)
    k_keep = min(k, n_elem)
    if kr == 0 or k_keep == 0:
        return th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    top_z = th.argsort(S, dim=1, descending=True, stable=True)[:, :k_keep]  # (B, K)
    s_z = th.gather(S, 1, top_z)  # (B, K)
    valid_z = s_z > _EMPTY_DEPTH_FLOOR  # (B, K)
    k_eff = max(
        1,
        int(np.count_nonzero(valid_z.detach().cpu().numpy().any(axis=0))),
    )
    if k_eff < k_keep:
        top_z = top_z[:, :k_eff]
        s_z = s_z[:, :k_eff]
        valid_z = valid_z[:, :k_eff]
        k_keep = k_eff
    has_valid_z = valid_z.any(dim=1, keepdim=True)  # (B, 1)
    z_pos = all_pos[top_z].to(dtype=dtype)  # (B, K, d)

    r_cen = rel_reg.cen.to(dtype=dtype)  # (B, K_r, d)
    r_w = rel_reg.w.to(dtype=dtype)  # (B, K_r, d)
    r_act = rel_reg.active  # (B, K_r)
    has_r_act = r_act.any(dim=1)  # (B,)

    if not th.is_grad_enabled() and device.type == "cpu":
        cen_zm_all = z_pos.unsqueeze(2) + r_cen.unsqueeze(1)  # (B, K, K_r, d)
        all_r_act = bool(r_act.detach().cpu().numpy().all())
        all_z_valid = bool(valid_z.detach().cpu().numpy().all())
        c_b = max(1, min(bsz, (1 << 16) // max(1, k_keep * kr * dim)))
        c_x_ng = max(
            1,
            min(
                n_elem,
                chunk,
                max_elems // max(1, bsz * k_keep * kr * dim),
                (1 << 20) // max(1, c_b * k_keep * kr * dim),
            ),
        )
        out = th.empty((bsz, n_elem), dtype=dtype, device=device)
        buf = th.empty(
            (c_b * k_keep * kr, c_x_ng, dim), dtype=dtype, device=device
        )
        sd_buf = th.empty((c_b, k_keep, kr, c_x_ng), dtype=dtype, device=device)
        dr_buf = th.empty((c_b, k_keep, c_x_ng), dtype=dtype, device=device)
        for b_start in range(0, bsz, c_b):
            n_b = min(c_b, bsz - b_start)
            m_b = n_b * k_keep * kr
            cen_f = (
                cen_zm_all[b_start : b_start + n_b]
                .reshape(m_b, 1, dim)
                .contiguous()
            )
            rw_f = (
                r_w[b_start : b_start + n_b, None, :, :]
                .expand(n_b, k_keep, kr, dim)
                .reshape(m_b, 1, dim)
                .contiguous()
            )
            ract_b = r_act[b_start : b_start + n_b]  # (n_b, K_r)
            hract_b = has_r_act[b_start : b_start + n_b]  # (n_b,)
            sz_b = s_z[b_start : b_start + n_b]  # (n_b, K)
            vz_b = valid_z[b_start : b_start + n_b]  # (n_b, K)
            hvz_b = has_valid_z[b_start : b_start + n_b]  # (n_b, 1)
            for x_start in range(0, n_elem, c_x_ng):
                n_x = min(c_x_ng, n_elem - x_start)
                p_chunk = (
                    all_pos[x_start : x_start + n_x]
                    .to(dtype=dtype)
                    .unsqueeze(0)
                )  # (1, n_x, d)
                b_sl = buf[:m_b, :n_x, :]
                sd_sl = sd_buf[:n_b, :, :, :n_x]
                dr_sl = dr_buf[:n_b, :, :n_x]
                th.sub(p_chunk, cen_f, out=b_sl)
                b_sl.abs_()
                th.sub(rw_f, b_sl, out=b_sl)
                th.amin(b_sl, dim=-1, out=sd_sl.view(m_b, n_x))
                if not all_r_act:
                    r_fill = th.where(
                        hract_b[:, None, None, None],
                        th.full_like(sd_sl, float("-inf")),
                        th.full_like(sd_sl, _EMPTY_DEPTH_FLOOR),
                    )
                    sd_sl.copy_(th.where(ract_b[:, None, :, None], sd_sl, r_fill))
                th.amax(sd_sl, dim=2, out=dr_sl)
                th.minimum(dr_sl, sz_b.unsqueeze(2), out=dr_sl)
                if not all_z_valid:
                    z_fill = th.where(
                        hvz_b.unsqueeze(2),
                        th.full_like(dr_sl, float("-inf")),
                        th.full_like(dr_sl, _EMPTY_DEPTH_FLOOR),
                    )
                    dr_sl.copy_(th.where(vz_b.unsqueeze(2), dr_sl, z_fill))
                th.amax(
                    dr_sl,
                    dim=1,
                    out=out[b_start : b_start + n_b, x_start : x_start + n_x],
                )
        return out

    c_x = min(n_elem, chunk)
    per_x = max(1, bsz * k_keep * kr * dim)
    if c_x * per_x > max_elems:
        c_x = max(1, min(c_x, max_elems // per_x))
    per_k = max(1, bsz * c_x * kr * dim)
    c_k = max(1, min(k_keep, max_elems // per_k)) if k_keep * per_k > max_elems else k_keep

    out_chunks: List[th.Tensor] = []
    for x_start in range(0, n_elem, c_x):
        p_chunk = all_pos[x_start : x_start + c_x].to(dtype=dtype)  # (n_x, d)
        best_x: Optional[th.Tensor] = None
        for k_start in range(0, k_keep, c_k):
            z_pos_k = z_pos[:, k_start : k_start + c_k, :]  # (B, n_k, d)
            s_z_k = s_z[:, k_start : k_start + c_k]  # (B, n_k)
            valid_z_k = valid_z[:, k_start : k_start + c_k]  # (B, n_k)

            cen_zm = z_pos_k.unsqueeze(2) + r_cen.unsqueeze(1)  # (B, n_k, K_r, d)
            coord_dep = r_w[:, None, None, :, :] - th.abs(
                p_chunk[None, :, None, None, :] - cen_zm[:, None, :, :, :]
            )  # (B, n_x, n_k, K_r, d)
            slot_dep = th.amin(coord_dep, dim=-1)  # (B, n_x, n_k, K_r)

            r_act_mask = r_act[:, None, None, :]  # (B, 1, 1, K_r)
            r_inactive_fill = th.where(
                has_r_act[:, None, None, None],
                th.full_like(slot_dep, float("-inf")),
                th.full_like(slot_dep, _EMPTY_DEPTH_FLOOR),
            )
            dep_r = th.amax(
                th.where(r_act_mask, slot_dep, r_inactive_fill), dim=-1
            )  # (B, n_x, n_k)

            cand_val = th.minimum(s_z_k.unsqueeze(1), dep_r)  # (B, n_x, n_k)
            z_inactive_fill = th.where(
                has_valid_z.unsqueeze(-1),
                th.full_like(cand_val, float("-inf")),
                th.full_like(cand_val, _EMPTY_DEPTH_FLOOR),
            )
            masked_val = th.where(valid_z_k.unsqueeze(1), cand_val, z_inactive_fill)
            chunk_k_best = th.amax(masked_val, dim=-1)  # (B, n_x)
            best_x = chunk_k_best if best_x is None else th.maximum(best_x, chunk_k_best)

        assert best_x is not None
        out_chunks.append(best_x)

    return out_chunks[0] if len(out_chunks) == 1 else th.cat(out_chunks, dim=1)


def _beam_project_type_rel(
    module: GroundedELModule,
    S: th.Tensor,
    *,
    k: int,
    all_pos: th.Tensor,
    max_elems: int = 1 << 26,
) -> th.Tensor:
    """Execute the ``type_rel`` step on beam scores ``S``: class elements get ``max_z min(S(z), depth(p_z, C))``, individuals get ``NEG``."""
    bsz, n_elem = S.shape
    n_cls = module.nb_classes
    dim = module.embed_dim
    dtype = S.dtype
    device = S.device

    if bsz == 0 or n_elem == 0:
        return th.empty((bsz, n_elem), dtype=dtype, device=device)

    k_keep = min(k, n_elem)
    cls_mb = class_boxes(module, None)  # (C, M_c, d)
    m_c = cls_mb.cen.shape[1]
    if m_c == 0 or k_keep == 0 or n_cls == 0:
        return th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    top_z = th.argsort(S, dim=1, descending=True, stable=True)[:, :k_keep]  # (B, K)
    s_z = th.gather(S, 1, top_z)  # (B, K)
    valid_z = s_z > _EMPTY_DEPTH_FLOOR  # (B, K)
    k_eff = max(
        1,
        int(np.count_nonzero(valid_z.detach().cpu().numpy().any(axis=0))),
    )
    if k_eff < k_keep:
        top_z = top_z[:, :k_eff]
        s_z = s_z[:, :k_eff]
        valid_z = valid_z[:, :k_eff]
        k_keep = k_eff
    has_valid_z = valid_z.any(dim=1, keepdim=True)  # (B, 1)
    z_pos = all_pos[top_z].to(dtype=dtype)  # (B, K, d)

    per_cls = max(1, bsz * k_keep * m_c * dim)
    c_chunk = max(1, min(n_cls, max_elems // per_cls))

    cls_chunks: List[th.Tensor] = []
    for c_start in range(0, n_cls, c_chunk):
        c_cen = cls_mb.cen[c_start : c_start + c_chunk].to(dtype=dtype)  # (n_c, M_c, d)
        c_w = cls_mb.w[c_start : c_start + c_chunk].to(dtype=dtype)  # (n_c, M_c, d)
        c_act = cls_mb.active[c_start : c_start + c_chunk]  # (n_c, M_c)
        c_has_act = c_act.any(dim=1)  # (n_c,)

        coord_dep = c_w[None, :, None, :, :] - th.abs(
            z_pos[:, None, :, None, :] - c_cen[None, :, None, :, :]
        )  # (B, n_c, K, M_c, d)
        slot_dep = th.amin(coord_dep, dim=-1)  # (B, n_c, K, M_c)

        c_act_mask = c_act[None, :, None, :]  # (1, n_c, 1, M_c)
        c_inactive_fill = th.where(
            c_has_act[None, :, None, None],
            th.full_like(slot_dep, float("-inf")),
            th.full_like(slot_dep, _EMPTY_DEPTH_FLOOR),
        )
        dep_zc = th.amax(
            th.where(c_act_mask, slot_dep, c_inactive_fill), dim=-1
        )  # (B, n_c, K)

        cand_val = th.minimum(s_z.unsqueeze(1), dep_zc)  # (B, n_c, K)
        z_inactive_fill = th.where(
            has_valid_z.unsqueeze(-1),
            th.full_like(cand_val, float("-inf")),
            th.full_like(cand_val, _EMPTY_DEPTH_FLOOR),
        )
        masked_val = th.where(valid_z.unsqueeze(1), cand_val, z_inactive_fill)
        cls_chunks.append(th.amax(masked_val, dim=-1))  # (B, n_c)

    cls_scores = cls_chunks[0] if len(cls_chunks) == 1 else th.cat(cls_chunks, dim=1)
    if n_elem > n_cls:
        ind_pad = th.full(
            (bsz, n_elem - n_cls), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device
        )
        return th.cat([cls_scores, ind_pad], dim=1)
    return cls_scores


def _beam_project_type_inv_rel(
    module: GroundedELModule,
    S: th.Tensor,
    *,
    k: int,
    chunk: int,
) -> th.Tensor:
    """Execute a non-initial ``type_inv_rel`` step on beam scores ``S`` ``(B, E)``.

    Only class elements (element id ``< module.nb_classes``) can be sources. With
    :math:`Z_C` the top-``k`` class elements of ``S[b, :nb_classes]`` (ties broken
    by smaller class id) and :math:`\\operatorname{dep}(p_y, c)` the crisp
    ``score_entities(module, class_boxes(module, c))`` depth of element :math:`y`,
    returns :math:`S'(y) = \\max_{c \\in Z_C} \\min(S(c), \\operatorname{dep}(p_y, c))`
    of shape ``(B, E)``.

    Guarantees:
    - Rows whose kept class scores are all at the empty floor (or modules with no
      classes) receive the empty floor ``NEG`` for every element.
    - With ``k >= module.nb_classes`` the result is the exact max-min combination
      over all classes; scores are monotone non-decreasing in ``k``.
    - Each row is computed independently of the other rows in the batch.

    Non-guarantees:
    - Gradients do not flow through the discrete top-``k`` class selection.
    """
    bsz, n_elem = S.shape
    n_cls = min(int(module.nb_classes), n_elem)
    dtype = S.dtype
    device = S.device
    if bsz == 0 or n_elem == 0:
        return th.empty((bsz, n_elem), dtype=dtype, device=device)
    k_keep = min(k, n_cls)
    if k_keep == 0:
        return th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    S_cls = S[:, :n_cls]
    top_c = th.argsort(S_cls, dim=1, descending=True, stable=True)[:, :k_keep]  # (B, K)
    s_c = th.gather(S_cls, 1, top_c)  # (B, K)
    valid_c = s_c > _EMPTY_DEPTH_FLOOR  # (B, K)
    k_eff = max(
        1,
        int(np.count_nonzero(valid_c.detach().cpu().numpy().any(axis=0))),
    )
    has_valid_c = valid_c.any(dim=1, keepdim=True)  # (B, 1)

    best = th.where(
        has_valid_c,
        th.full((bsz, n_elem), float("-inf"), dtype=dtype, device=device),
        th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device),
    )
    for j in range(k_eff):
        dep_j = score_entities(
            module, class_boxes(module, top_c[:, j]), chunk=chunk
        ).to(dtype=dtype)  # (B, E)
        cand_j = th.minimum(s_c[:, j : j + 1], dep_j)
        cand_j = th.where(
            valid_c[:, j : j + 1], cand_j, th.full_like(cand_j, float("-inf"))
        )
        best = th.maximum(best, cand_j)
    return best


def _apply_beam_hop(
    module: GroundedELModule,
    S: th.Tensor,
    hop_rels_tuple: Tuple[int, ...],
    *,
    is_first_hop: bool,
    anchor_cls_ids: Optional[th.Tensor],
    all_anchors_are_classes: bool,
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    k: int,
    chunk: int,
    all_pos: th.Tensor,
) -> th.Tensor:
    """Apply a single relation hop to element score matrix ``S`` ``(B, E)`` without Python loops over the batch."""
    device = S.device
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
    # ``type_inv_rel`` after the first hop: sources are the class elements of ``S``
    has_type_inv_later = (
        (not is_first_hop)
        and type_inv_rel is not None
        and (type_inv_rel in rel_set)
    )
    only_type_inv_later = has_type_inv_later and rel_set == {type_inv_rel}
    only_normal = (not has_type_inv) and (not has_type) and (not has_type_inv_later)

    if only_type_inv:
        assert anchor_cls_ids is not None
        return score_entities(
            module, class_boxes(module, anchor_cls_ids), chunk=chunk
        )

    if only_type_inv_later:
        return _beam_project_type_inv_rel(module, S, k=k, chunk=chunk)

    if only_type:
        return _beam_project_type_rel(module, S, k=k, all_pos=all_pos)

    if only_normal:
        return _beam_project_rel(
            module, S, r_t, k=k, chunk=chunk, all_pos=all_pos
        )

    special_rels = set()
    if has_type_inv and all_anchors_are_classes and type_inv_rel is not None:
        special_rels.add(type_inv_rel)
    if has_type_inv_later and type_inv_rel is not None:
        special_rels.add(type_inv_rel)
    if has_type and type_rel is not None:
        special_rels.add(type_rel)
    has_normal = len(rel_set - special_rels) > 0

    cur: Optional[th.Tensor] = None
    if has_normal:
        safe_r = r_t
        if type_inv_rel is not None:
            safe_r = th.where(safe_r == type_inv_rel, th.zeros_like(safe_r), safe_r)
        if type_rel is not None:
            safe_r = th.where(safe_r == type_rel, th.zeros_like(safe_r), safe_r)
        cur = _beam_project_rel(
            module, S, safe_r, k=k, chunk=chunk, all_pos=all_pos
        )

    if has_type:
        s_type = _beam_project_type_rel(module, S, k=k, all_pos=all_pos)
        if cur is None:
            cur = s_type
        else:
            t_mask = (r_t == type_rel).unsqueeze(1)
            cur = th.where(t_mask, s_type, cur)

    if has_type_inv_later:
        s_inv_later = _beam_project_type_inv_rel(module, S, k=k, chunk=chunk)
        if cur is None:
            cur = s_inv_later
        else:
            il_mask = (r_t == type_inv_rel).unsqueeze(1)
            cur = th.where(il_mask, s_inv_later, cur)

    if has_type_inv:
        assert anchor_cls_ids is not None
        inv_mask = (r_t == type_inv_rel) & (anchor_cls_ids >= 0)
        safe_cls = th.clamp_min(anchor_cls_ids, 0)
        s_inv = score_entities(
            module, class_boxes(module, safe_cls), chunk=chunk
        )
        if cur is None:
            cur = th.where(
                inv_mask.unsqueeze(1),
                s_inv,
                th.full_like(s_inv, _EMPTY_DEPTH_FLOOR),
            )
        else:
            cur = th.where(inv_mask.unsqueeze(1), s_inv, cur)

    assert cur is not None
    return cur


def _score_beam_recursive(
    module: GroundedELModule,
    structure: Any,
    queries: Sequence[Any],
    *,
    e2el_t: Optional[th.Tensor],
    c2e_t: Optional[th.Tensor],
    c2e_arr: Optional[np.ndarray],
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    k: int,
    chunk: int,
    all_pos: th.Tensor,
) -> th.Tensor:
    """Recursively compute ``(B, E)`` beam scores for ``queries`` matching ``structure``."""
    bsz = len(queries)
    n_elem = module.nb_elements
    device = all_pos.device
    dtype = all_pos.dtype

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
            S = th.full(
                (bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device
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
            if safe_elem_ids.numel() > 0 and bool(
                ((safe_elem_ids < 0) | (safe_elem_ids >= n_elem)).any()
            ):
                raise ValueError(
                    f"Anchor element id out of range [0, {n_elem})"
                )
            elem_grid = th.arange(n_elem, dtype=th.long, device=device).unsqueeze(0)
            S = th.where(
                elem_grid == safe_elem_ids.unsqueeze(1),
                th.full((bsz, n_elem), _TOP_DEPTH_CEIL, dtype=dtype, device=device),
                th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device),
            )

        for hop in range(n_hops):
            S = _apply_beam_hop(
                module,
                S,
                hop_tuples[hop],
                is_first_hop=(hop == 0),
                anchor_cls_ids=anchor_cls_ids if hop == 0 else None,
                all_anchors_are_classes=all_anchors_are_classes if hop == 0 else False,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                k=k,
                chunk=chunk,
                all_pos=all_pos,
            )
        return S

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
        S = _score_beam_recursive(
            module,
            sub_struct,
            sub_queries,
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            k=k,
            chunk=chunk,
            all_pos=all_pos,
        )
        for hop in range(n_hops):
            S = _apply_beam_hop(
                module,
                S,
                hop_tuples[hop],
                is_first_hop=False,
                anchor_cls_ids=None,
                all_anchors_are_classes=False,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                k=k,
                chunk=chunk,
                all_pos=all_pos,
            )
        return S

    # Case 3: Union (sub_1, sub_2, ..., ('u',))
    if structure[-1] == ("u",):
        if len(structure) < 3:
            raise ValueError(
                f"Union structure must have at least 2 disjuncts before ('u',), got {structure!r}"
            )
        n_disj = len(structure) - 1
        if len(cols) < n_disj:
            raise ValueError(f"Queries do not match union structure {structure!r}")
        S_acc = _score_beam_recursive(
            module,
            structure[0],
            cols[0],
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            k=k,
            chunk=chunk,
            all_pos=all_pos,
        )
        for i in range(1, n_disj):
            S_i = _score_beam_recursive(
                module,
                structure[i],
                cols[i],
                e2el_t=e2el_t,
                c2e_t=c2e_t,
                c2e_arr=c2e_arr,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                k=k,
                chunk=chunk,
                all_pos=all_pos,
            )
            S_acc = th.maximum(S_acc, S_i)
        return S_acc

    # Case 4: Intersection (sub_1, sub_2, ...)
    n_conj = len(structure)
    if len(cols) != n_conj:
        raise ValueError(f"Queries do not match intersection structure {structure!r}")
    S_acc = _score_beam_recursive(
        module,
        structure[0],
        cols[0],
        e2el_t=e2el_t,
        c2e_t=c2e_t,
        c2e_arr=c2e_arr,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        k=k,
        chunk=chunk,
        all_pos=all_pos,
    )
    for i in range(1, n_conj):
        S_i = _score_beam_recursive(
            module,
            structure[i],
            cols[i],
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            k=k,
            chunk=chunk,
            all_pos=all_pos,
        )
        S_acc = th.minimum(S_acc, S_i)
    return S_acc


def score_query_beam(
    module: GroundedELModule,
    structure: Any,
    queries: Sequence[Any],
    *,
    k: int = 64,
    type_rel: Optional[int] = None,
    type_inv_rel: Optional[int] = None,
    class_of_entity: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    entity_to_element: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    candidate_element_ids: Optional[Union[int, Sequence[int], np.ndarray, th.Tensor]] = None,
    chunk: int = 65536,
) -> th.Tensor:
    """Score a batch of Query2Box-structured ``queries`` via recursive top-``k`` max-min (Gödel t-norm) beam search.

    Maintains a score tensor :math:`S` of shape ``(B, E)`` over all domain elements
    :math:`E = \\text{module.nb\\_elements}` (with unreachable floor ``NEG = -1e6``
    and anchor ceiling ``BIG = +1e6``):

    - **Anchor** :math:`e`: :math:`S[b, \\text{el}(e)] = +10^6` and :math:`-10^6`
      elsewhere.
    - **Projection** by relation :math:`r` from :math:`S`: selects the top-``k``
      element indices :math:`Z` of :math:`S` per query (ties broken by smaller
      element index) and evaluates
      :math:`S'(x) = \\max_{z \\in Z} \\min(S(z),\\, \\operatorname{dep}_r(z, x))`,
      where :math:`\\operatorname{dep}_r(z, x) = \\max_m \\min_j (w_{r,m,j} - |p_{x,j} - p_{z,j} - c_{r,m,j}|)`
      over active translation boxes :math:`m` of :math:`r` and entries with
      :math:`S(z) \\le \\text{NEG}` contribute ``NEG``.
    - **``type_inv_rel``** from a class-entity anchor :math:`A`:
      :math:`S'(x) = \\operatorname{depth}(p_x,\\, \\text{class\\_boxes}(A))`.
    - **``type_inv_rel``** at a non-first hop from :math:`S`: with :math:`Z_C` the
      top-``k`` class elements (:math:`\\text{id} < \\text{nb\\_classes}`) of
      :math:`S`, :math:`S'(x) = \\max_{c \\in Z_C} \\min(S(c),\\, \\operatorname{depth}(p_x,\\, \\text{class\\_boxes}(c)))`;
      individual elements of :math:`S` are never sources.
    - **``type_rel``** from :math:`S`: for every class :math:`c < \\text{nb\\_classes}`,
      :math:`S'(c) = \\max_{z \\in Z} \\min(S(z),\\, \\operatorname{depth}(p_z,\\, \\text{class\\_boxes}(c)))`;
      all individual elements (:math:`\\text{id} \\ge \\text{nb\\_classes}`) receive ``NEG``.
    - **Intersection / Union:** elementwise ``min`` / ``max`` of branch score tensors.

    Guarantees:
    - **Reasoner-free (R1):** Contains no imports or calls to reasoners or semantic
      completion routines.
    - **1p equivalence:** On single-hop queries ``('e', ('r',))`` (including
      ``type_inv_rel`` class-entity anchors), ``score_query_beam`` equals
      ``score_entities(module, compile_query(module, ('e', ('r',)), ...))`` for any ``k >= 1``.
    - **Crisp consistency:** With ``k >= module.nb_elements``, the crisp answer set
      :math:`\\{x : S(x) \\ge 0\\}` (strict interior :math:`\\{x : S(x) > 0\\}` off
      measure-zero box boundaries) equals the element-semantics answer set of
      :func:`project_exact` with unlimited witnesses and boxes.
    - **Monotonicity in ``k``:** For any single-hop or two-stage query structure
      (``1p``, ``2p``, ``2i``, ``3i``, ``2u``, ``ip``, ``pi``, ``up``), scores are
      monotone non-decreasing in ``k`` (``k_1 <= k_2`` implies
      ``score_query_beam(..., k=k_1) <= score_query_beam(..., k=k_2)`` elementwise),
      including hops with ``type_rel`` / ``type_inv_rel``.
    - **Mixed special relations:** A hop whose batch mixes ``type_inv_rel``,
      ``type_rel`` and ordinary roles is evaluated row-wise; each row equals the
      score obtained by scoring that row alone.
    - **Memory-bounded and differentiable:** Evaluates projections in chunks of at
      most ``chunk`` elements without calling ``.item()``, remaining ``no_grad``-safe
      and differentiable when gradients are enabled.

    Non-guarantees:
    - When ``k < module.nb_elements``, witnesses outside the top-``k`` elements of
      :math:`S` at an intermediate projection step are pruned.
    - Gradients flow through the selected top-``k`` witness paths and winning
      box/coordinate indices, not through the discrete ``argsort`` witness selection.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module in ``"translation"`` role mode.
    structure : Any
        Query2Box structure tuple describing the query syntax tree.
    queries : Sequence[Any]
        Batch of query instances matching ``structure``.
    k : int, default 64
        Beam width (number of top-scoring witness elements retained per projection step) :math:`\\ge 1`.
    type_rel : Optional[int], default None
        Optional relation id representing ``rdf:type`` (concept-level step).
    type_inv_rel : Optional[int], default None
        Optional relation id representing ``rdf:type^-`` (class-instance step).
    class_of_entity : Optional[Union[Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to class ids in
        ``[0, module.nb_classes)`` (or ``-1`` for non-class entities).
    entity_to_element : Optional[Union[Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to module element ids in
        ``[0, module.nb_elements)``. When ``None``, anchor ids in ``queries``
        are used directly as module element ids.
    candidate_element_ids : Optional[Union[int, Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D candidate element indices in ``[0, module.nb_elements)`` to
        gather at the end, or ``None`` for all elements ``0 .. module.nb_elements - 1``.
    chunk : int, default 65536
        Element chunk size :math:`\\ge 1` for projection steps.

    Returns
    -------
    torch.Tensor
        Candidate element scores of shape ``(B, N)`` where ``N = module.nb_elements``
        when ``candidate_element_ids is None``.
    """
    _validate_module(module, require_translation=True)
    k_val = _validate_int_ge("k", k, 1)
    chunk_sz = _validate_int_ge("chunk", chunk, 1)
    if not isinstance(queries, (list, tuple)):
        raise TypeError(
            f"queries must be a Sequence (list or tuple) of query instances, got {type(queries).__name__}"
        )

    device = module.elem_pos.weight.device
    dtype = module.elem_pos.weight.dtype
    n_elem = module.nb_elements

    cand_idx: Optional[th.Tensor] = None
    if candidate_element_ids is not None:
        cand_idx = _to_1d_long(candidate_element_ids, device, "candidate_element_ids")
        if cand_idx.numel() > 0 and bool(
            ((cand_idx < 0) | (cand_idx >= n_elem)).any()
        ):
            raise ValueError(
                f"candidate_element_ids indices out of range [0, {n_elem})"
            )

    if len(queries) == 0:
        n_out = n_elem if cand_idx is None else int(cand_idx.shape[0])
        return th.empty((0, n_out), dtype=dtype, device=device)

    e2el_t: Optional[th.Tensor] = None
    if entity_to_element is not None:
        e2el_t = _to_1d_long(entity_to_element, device, "entity_to_element")

    c2e_t: Optional[th.Tensor] = None
    c2e_arr: Optional[np.ndarray] = None
    if class_of_entity is not None:
        c2e_t = _to_1d_long(class_of_entity, device, "class_of_entity")
        c2e_arr = c2e_t.detach().cpu().numpy()

    all_pos = module.positions()  # (E, d)
    S = _score_beam_recursive(
        module,
        structure,
        queries,
        e2el_t=e2el_t,
        c2e_t=c2e_t,
        c2e_arr=c2e_arr,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        k=k_val,
        chunk=chunk_sz,
        all_pos=all_pos,
    )
    if cand_idx is None:
        return S
    return S[:, cand_idx]


def _validate_finite_margin(name: str, val: Union[float, int, th.Tensor]) -> float:
    """Validate that ``val`` is a finite scalar float without calling ``.item()``."""
    if isinstance(val, th.Tensor):
        if (
            val.dtype == th.bool
            or val.is_complex()
            or val.numel() != 1
        ):
            raise ValueError(f"{name} must be a finite scalar, got {val!r}")
        v = float(val.detach().cpu().numpy().flat[0])
    elif isinstance(val, (bool, np.bool_)) or not isinstance(
        val, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    else:
        v = float(val)
    if np.isnan(v) or np.isinf(v):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return v


def _to_2d_long_targets(
    ids: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, th.Tensor],
    bsz: int,
    n_elem: int,
    device: th.device,
    name: str,
    allow_empty_cols: bool = False,
) -> Tuple[th.Tensor, bool]:
    """Convert ``ids`` to a 2-D ``torch.long`` tensor of shape ``(B, M)`` in ``[0, n_elem)`` and return ``(tensor, was_1d)``."""
    if isinstance(ids, (int, np.integer)) and not isinstance(ids, (bool, np.bool_)):
        idx = th.tensor([[int(ids)]], dtype=th.long, device=device)
        was_1d = True
    elif isinstance(ids, th.Tensor):
        if ids.dtype == th.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError(f"{name} must contain integer indices, got dtype={ids.dtype}")
        idx = ids.to(device=device, dtype=th.long)
        if idx.ndim == 0:
            idx = idx.view(1, 1)
            was_1d = True
        elif idx.ndim == 1:
            idx = idx.unsqueeze(1)
            was_1d = True
        elif idx.ndim == 2:
            was_1d = False
        else:
            raise ValueError(
                f"{name} must be 1-D (B,) or 2-D (B, M), got shape {tuple(ids.shape)}"
            )
    elif isinstance(ids, (list, tuple, np.ndarray)):
        arr = np.asarray(ids)
        if arr.size > 0 and (
            arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer)
        ):
            raise ValueError(f"{name} must contain integer indices, got dtype={arr.dtype}")
        if arr.ndim == 0:
            idx = th.as_tensor(arr, dtype=th.long, device=device).view(1, 1)
            was_1d = True
        elif arr.ndim == 1:
            if bsz == 0 and arr.shape[0] == 0:
                idx = th.empty((0, 1), dtype=th.long, device=device)
            else:
                idx = th.as_tensor(arr, dtype=th.long, device=device).unsqueeze(1)
            was_1d = True
        elif arr.ndim == 2:
            idx = th.as_tensor(arr, dtype=th.long, device=device)
            was_1d = False
        else:
            raise ValueError(
                f"{name} must be 1-D (B,) or 2-D (B, M), got shape {arr.shape}"
            )
    else:
        raise TypeError(
            f"{name} must be an int, Sequence, ndarray, or Tensor, got {type(ids).__name__}"
        )

    if idx.shape[0] != bsz:
        raise ValueError(
            f"{name} batch size ({idx.shape[0]}) does not match queries batch size ({bsz})"
        )
    if not allow_empty_cols and bsz > 0 and idx.shape[1] < 1:
        raise ValueError(
            f"{name} must have at least 1 column per query, got shape {tuple(idx.shape)}"
        )
    if idx.numel() > 0 and bool(((idx < 0) | (idx >= n_elem)).any()):
        raise ValueError(f"{name} indices out of range [0, {n_elem})")
    return idx, was_1d


def _soft_agg_slot_depth(
    slot_depth: th.Tensor,
    active: th.Tensor,
    has_active: th.Tensor,
    tau: th.Tensor,
) -> th.Tensor:
    """Aggregate per-component signed depth ``(..., M_slots)`` over ``active`` slots with temperature ``tau``.

    Uses the numerically shifted logsumexp identity
    ``m_det + tau * logsumexp((slot_depth - m_det) / tau)`` where ``m_det`` is the
    detached crisp maximum over active slots, so that at any ``tau > 0`` the value
    and gradient equal ``tau * logsumexp(slot_depth / tau)`` while remaining
    overflow-free and converging to the exact crisp maximum as ``tau -> 0+``.
    """
    if slot_depth.shape[-1] == 0:
        return th.full(
            slot_depth.shape[:-1],
            _EMPTY_DEPTH_FLOOR,
            dtype=slot_depth.dtype,
            device=slot_depth.device,
        )
    inactive_fill = th.where(
        has_active,
        th.full_like(slot_depth, float("-inf")),
        th.full_like(slot_depth, _EMPTY_DEPTH_FLOOR),
    )
    masked = th.where(active, slot_depth, inactive_fill)
    max_val = th.amax(masked, dim=-1, keepdim=True)
    max_det = th.where(
        has_active,
        max_val.detach(),
        th.zeros_like(max_val),
    )
    tau_t = tau.to(dtype=slot_depth.dtype, device=slot_depth.device)
    shifted = th.where(
        has_active,
        (masked - max_det) / tau_t,
        th.zeros_like(masked),
    )
    lse = max_det.squeeze(-1) + tau_t * th.logsumexp(shifted, dim=-1)
    return th.where(
        has_active.squeeze(-1),
        lse,
        th.full_like(lse, _EMPTY_DEPTH_FLOOR),
    )


def _soft_score_class_targets(
    module: GroundedELModule,
    cls_ids: th.Tensor,
    target_ids: th.Tensor,
    *,
    all_pos: th.Tensor,
    tau: th.Tensor,
) -> th.Tensor:
    """Compute soft depth of ``target_ids`` ``(B, M)`` in ``class_boxes(module, cls_ids)`` ``(B,)``."""
    bsz, m_targ = target_ids.shape
    cls_mb = class_boxes(module, cls_ids)  # (B, K_c, d)
    dtype = cls_mb.cen.dtype
    device = cls_mb.cen.device
    if bsz == 0 or m_targ == 0:
        return th.empty((bsz, m_targ), dtype=dtype, device=device)
    if cls_mb.cen.shape[1] == 0:
        return th.full((bsz, m_targ), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    t_pos = all_pos[target_ids].to(dtype=dtype)  # (B, M, d)
    coord_dep = cls_mb.w.unsqueeze(1) - th.abs(
        t_pos.unsqueeze(2) - cls_mb.cen.unsqueeze(1)
    )  # (B, M, K_c, d)
    slot_dep = th.amin(coord_dep, dim=-1)  # (B, M, K_c)
    act_mask = cls_mb.active.unsqueeze(1)  # (B, 1, K_c)
    has_act = act_mask.any(dim=-1, keepdim=True)  # (B, 1, 1)
    return _soft_agg_slot_depth(slot_dep, act_mask, has_act, tau)


def _soft_project_rel_to_targets(
    module: GroundedELModule,
    z_pos: th.Tensor,
    s_z: th.Tensor,
    valid_z: th.Tensor,
    rel_ids: th.Tensor,
    target_ids: th.Tensor,
    *,
    all_pos: th.Tensor,
    tau: th.Tensor,
    max_elems: int = 1 << 22,
) -> th.Tensor:
    """Project witness scores ``s_z`` ``(B, K)`` at ``z_pos`` ``(B, K, d)`` along ``rel_ids`` to ``target_ids`` ``(B, M)`` using soft depth."""
    bsz, m_targ = target_ids.shape
    k_wit = z_pos.shape[1]
    dim = module.embed_dim
    dtype = z_pos.dtype
    device = z_pos.device

    rel_reg = module.translation_regions(rel_ids)
    kr = rel_reg.cen.shape[1]
    if bsz == 0 or m_targ == 0:
        return th.empty((bsz, m_targ), dtype=dtype, device=device)
    if kr == 0 or k_wit == 0:
        return th.full((bsz, m_targ), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    has_valid_z = valid_z.any(dim=1, keepdim=True)  # (B, 1)
    r_cen = rel_reg.cen.to(dtype=dtype)  # (B, K_r, d)
    r_w = rel_reg.w.to(dtype=dtype)  # (B, K_r, d)
    r_act = rel_reg.active  # (B, K_r)
    r_act_mask = r_act[:, None, None, :]  # (B, 1, 1, K_r)
    has_r_act = r_act_mask.any(dim=-1, keepdim=True)  # (B, 1, 1, 1)
    cen_zm = z_pos.unsqueeze(2) + r_cen.unsqueeze(1)  # (B, K, K_r, d)

    per_m = max(1, bsz * k_wit * kr * dim)
    c_m = max(1, min(m_targ, max_elems // per_m))

    out_chunks: List[th.Tensor] = []
    for m_start in range(0, m_targ, c_m):
        t_sub = target_ids[:, m_start : m_start + c_m]  # (B, n_m)
        p_targ = all_pos[t_sub].to(dtype=dtype)  # (B, n_m, d)
        coord_dep = r_w[:, None, None, :, :] - th.abs(
            p_targ[:, :, None, None, :] - cen_zm[:, None, :, :, :]
        )  # (B, n_m, K, K_r, d)
        slot_dep = th.amin(coord_dep, dim=-1)  # (B, n_m, K, K_r)
        dep_r = _soft_agg_slot_depth(slot_dep, r_act_mask, has_r_act, tau)  # (B, n_m, K)

        cand_val = th.minimum(s_z.unsqueeze(1), dep_r)  # (B, n_m, K)
        z_inactive_fill = th.where(
            has_valid_z.unsqueeze(-1),
            th.full_like(cand_val, float("-inf")),
            th.full_like(cand_val, _EMPTY_DEPTH_FLOOR),
        )
        masked_val = th.where(valid_z.unsqueeze(1), cand_val, z_inactive_fill)
        out_chunks.append(th.amax(masked_val, dim=-1))  # (B, n_m)

    return out_chunks[0] if len(out_chunks) == 1 else th.cat(out_chunks, dim=1)


def _soft_project_type_rel_to_targets(
    module: GroundedELModule,
    z_pos: th.Tensor,
    s_z: th.Tensor,
    valid_z: th.Tensor,
    target_ids: th.Tensor,
    *,
    tau: th.Tensor,
    max_elems: int = 1 << 22,
) -> th.Tensor:
    """Execute the ``type_rel`` step from witnesses ``(z_pos, s_z, valid_z)`` to ``target_ids`` ``(B, M)`` using soft depth."""
    bsz, m_targ = target_ids.shape
    k_wit = z_pos.shape[1]
    n_cls = module.nb_classes
    dim = module.embed_dim
    dtype = z_pos.dtype
    device = z_pos.device

    if bsz == 0 or m_targ == 0:
        return th.empty((bsz, m_targ), dtype=dtype, device=device)

    cls_mb = class_boxes(module, None)  # (C, M_c, d)
    m_c = cls_mb.cen.shape[1]
    if m_c == 0 or k_wit == 0 or n_cls == 0:
        return th.full((bsz, m_targ), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    is_cls_targ = (target_ids >= 0) & (target_ids < n_cls)  # (B, M)
    if not bool(is_cls_targ.detach().cpu().numpy().any()):
        return th.full((bsz, m_targ), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    has_valid_z = valid_z.any(dim=1, keepdim=True)  # (B, 1)
    safe_cls_ids = th.where(is_cls_targ, target_ids, th.zeros_like(target_ids))

    per_m = max(1, bsz * k_wit * m_c * dim)
    c_m = max(1, min(m_targ, max_elems // per_m))

    out_chunks: List[th.Tensor] = []
    for m_start in range(0, m_targ, c_m):
        c_sub = safe_cls_ids[:, m_start : m_start + c_m]  # (B, n_m)
        cls_mask_sub = is_cls_targ[:, m_start : m_start + c_m]  # (B, n_m)
        c_cen = cls_mb.cen[c_sub].to(dtype=dtype)  # (B, n_m, M_c, d)
        c_w = cls_mb.w[c_sub].to(dtype=dtype)  # (B, n_m, M_c, d)
        c_act = cls_mb.active[c_sub]  # (B, n_m, M_c)

        coord_dep = c_w[:, :, None, :, :] - th.abs(
            z_pos[:, None, :, None, :] - c_cen[:, :, None, :, :]
        )  # (B, n_m, K, M_c, d)
        slot_dep = th.amin(coord_dep, dim=-1)  # (B, n_m, K, M_c)
        c_act_mask = c_act[:, :, None, :]  # (B, n_m, 1, M_c)
        c_has_act = c_act_mask.any(dim=-1, keepdim=True)  # (B, n_m, 1, 1)
        dep_zc = _soft_agg_slot_depth(slot_dep, c_act_mask, c_has_act, tau)  # (B, n_m, K)

        cand_val = th.minimum(s_z.unsqueeze(1), dep_zc)  # (B, n_m, K)
        z_inactive_fill = th.where(
            has_valid_z.unsqueeze(-1),
            th.full_like(cand_val, float("-inf")),
            th.full_like(cand_val, _EMPTY_DEPTH_FLOOR),
        )
        masked_val = th.where(valid_z.unsqueeze(1), cand_val, z_inactive_fill)
        chunk_scores = th.amax(masked_val, dim=-1)  # (B, n_m)
        chunk_scores = th.where(
            cls_mask_sub,
            chunk_scores,
            th.full_like(chunk_scores, _EMPTY_DEPTH_FLOOR),
        )
        out_chunks.append(chunk_scores)

    return out_chunks[0] if len(out_chunks) == 1 else th.cat(out_chunks, dim=1)


def _soft_project_type_inv_rel_to_targets(
    module: GroundedELModule,
    top_z: th.Tensor,
    s_z: th.Tensor,
    valid_z: th.Tensor,
    target_ids: th.Tensor,
    *,
    all_pos: th.Tensor,
    tau: th.Tensor,
    max_elems: int = 1 << 22,
) -> th.Tensor:
    """Execute a non-initial ``type_inv_rel`` step from witnesses ``(top_z, s_z, valid_z)`` ``(B, K)`` to ``target_ids`` ``(B, M)``.

    Only witnesses that are class elements (``top_z < module.nb_classes``) act as
    sources; returns :math:`\\max_c \\min(s_z(c), \\operatorname{dep}_\\tau(p_t, \\text{class\\_boxes}(c)))`
    using the same soft class depth as :func:`_soft_score_class_targets`, and
    ``NEG`` for rows without a valid class witness.
    """
    bsz, m_targ = target_ids.shape
    k_wit = top_z.shape[1]
    n_cls = module.nb_classes
    dim = module.embed_dim
    dtype = s_z.dtype
    device = s_z.device

    if bsz == 0 or m_targ == 0:
        return th.empty((bsz, m_targ), dtype=dtype, device=device)
    if k_wit == 0 or n_cls == 0:
        return th.full((bsz, m_targ), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)

    is_cls_z = (top_z >= 0) & (top_z < n_cls)  # (B, K)
    valid_c = valid_z & is_cls_z  # (B, K)
    safe_c = th.where(is_cls_z, top_z, th.zeros_like(top_z))
    cls_mb = class_boxes(module, safe_c.reshape(-1))  # (B*K, M_c, d)
    m_c = cls_mb.cen.shape[1]
    if m_c == 0:
        return th.full((bsz, m_targ), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)
    c_cen = cls_mb.cen.to(dtype=dtype).view(bsz, k_wit, m_c, dim)
    c_w = cls_mb.w.to(dtype=dtype).view(bsz, k_wit, m_c, dim)
    c_act = cls_mb.active.view(bsz, k_wit, m_c)
    c_act_mask = c_act[:, None, :, :]  # (B, 1, K, M_c)
    c_has_act = c_act_mask.any(dim=-1, keepdim=True)  # (B, 1, K, 1)
    has_valid_c = valid_c.any(dim=1, keepdim=True)  # (B, 1)

    per_m = max(1, bsz * k_wit * m_c * dim)
    c_m = max(1, min(m_targ, max_elems // per_m))

    out_chunks: List[th.Tensor] = []
    for m_start in range(0, m_targ, c_m):
        t_sub = target_ids[:, m_start : m_start + c_m]  # (B, n_m)
        p_targ = all_pos[t_sub].to(dtype=dtype)  # (B, n_m, d)
        coord_dep = c_w[:, None, :, :, :] - th.abs(
            p_targ[:, :, None, None, :] - c_cen[:, None, :, :, :]
        )  # (B, n_m, K, M_c, d)
        slot_dep = th.amin(coord_dep, dim=-1)  # (B, n_m, K, M_c)
        dep_tc = _soft_agg_slot_depth(slot_dep, c_act_mask, c_has_act, tau)  # (B, n_m, K)

        cand_val = th.minimum(s_z.unsqueeze(1), dep_tc)  # (B, n_m, K)
        z_inactive_fill = th.where(
            has_valid_c.unsqueeze(-1),
            th.full_like(cand_val, float("-inf")),
            th.full_like(cand_val, _EMPTY_DEPTH_FLOOR),
        )
        masked_val = th.where(valid_c.unsqueeze(1), cand_val, z_inactive_fill)
        out_chunks.append(th.amax(masked_val, dim=-1))  # (B, n_m)

    return out_chunks[0] if len(out_chunks) == 1 else th.cat(out_chunks, dim=1)


def _apply_soft_hop_from_anchor(
    module: GroundedELModule,
    safe_elem_ids: th.Tensor,
    hop_rels_tuple: Tuple[int, ...],
    target_ids: th.Tensor,
    *,
    anchor_cls_ids: Optional[th.Tensor],
    all_anchors_are_classes: bool,
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    all_pos: th.Tensor,
    tau: th.Tensor,
) -> th.Tensor:
    """Evaluate the first hop from anchor ``safe_elem_ids`` ``(B,)`` at ``target_ids`` ``(B, M)`` using soft depth."""
    bsz = safe_elem_ids.shape[0]
    device = all_pos.device
    dtype = all_pos.dtype
    rel_set = set(hop_rels_tuple)
    r_t = _to_1d_long(hop_rels_tuple, device, "rel_ids")

    has_type_inv = (
        type_inv_rel is not None
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
        return _soft_score_class_targets(
            module, anchor_cls_ids, target_ids, all_pos=all_pos, tau=tau
        )

    z_pos = all_pos[safe_elem_ids].unsqueeze(1).to(dtype=dtype)  # (B, 1, d)
    s_z = th.full((bsz, 1), _TOP_DEPTH_CEIL, dtype=dtype, device=device)
    valid_z = th.ones((bsz, 1), dtype=th.bool, device=device)

    if only_type:
        return _soft_project_type_rel_to_targets(
            module, z_pos, s_z, valid_z, target_ids, tau=tau
        )

    if only_normal:
        return _soft_project_rel_to_targets(
            module, z_pos, s_z, valid_z, r_t, target_ids, all_pos=all_pos, tau=tau
        )

    special_rels = set()
    if has_type_inv and all_anchors_are_classes and type_inv_rel is not None:
        special_rels.add(type_inv_rel)
    if has_type and type_rel is not None:
        special_rels.add(type_rel)
    has_normal = len(rel_set - special_rels) > 0

    cur: Optional[th.Tensor] = None
    if has_normal:
        safe_r = r_t
        if type_inv_rel is not None:
            safe_r = th.where(safe_r == type_inv_rel, th.zeros_like(safe_r), safe_r)
        if type_rel is not None:
            safe_r = th.where(safe_r == type_rel, th.zeros_like(safe_r), safe_r)
        cur = _soft_project_rel_to_targets(
            module, z_pos, s_z, valid_z, safe_r, target_ids, all_pos=all_pos, tau=tau
        )

    if has_type:
        s_type = _soft_project_type_rel_to_targets(
            module, z_pos, s_z, valid_z, target_ids, tau=tau
        )
        if cur is None:
            cur = s_type
        else:
            t_mask = (r_t == type_rel).unsqueeze(1)
            cur = th.where(t_mask, s_type, cur)

    if has_type_inv:
        assert anchor_cls_ids is not None
        inv_mask = (r_t == type_inv_rel) & (anchor_cls_ids >= 0)
        safe_cls = th.clamp_min(anchor_cls_ids, 0)
        s_inv = _soft_score_class_targets(
            module, safe_cls, target_ids, all_pos=all_pos, tau=tau
        )
        if cur is None:
            cur = th.where(
                inv_mask.unsqueeze(1),
                s_inv,
                th.full_like(s_inv, _EMPTY_DEPTH_FLOOR),
            )
        else:
            cur = th.where(inv_mask.unsqueeze(1), s_inv, cur)

    assert cur is not None
    return cur


def _apply_soft_hop_from_witnesses(
    module: GroundedELModule,
    top_z: th.Tensor,
    s_z: th.Tensor,
    valid_z: th.Tensor,
    hop_rels_tuple: Tuple[int, ...],
    target_ids: th.Tensor,
    *,
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    all_pos: th.Tensor,
    tau: th.Tensor,
) -> th.Tensor:
    """Apply a non-initial relation hop from witnesses ``(top_z, s_z, valid_z)`` ``(B, K)`` to ``target_ids`` ``(B, M)`` using soft depth."""
    device = all_pos.device
    dtype = all_pos.dtype
    rel_set = set(hop_rels_tuple)
    r_t = _to_1d_long(hop_rels_tuple, device, "rel_ids")

    has_type = type_rel is not None and (type_rel in rel_set)
    only_type = has_type and rel_set == {type_rel}
    has_type_inv = type_inv_rel is not None and (type_inv_rel in rel_set)
    only_type_inv = has_type_inv and rel_set == {type_inv_rel}
    only_normal = (not has_type) and (not has_type_inv)

    z_pos = all_pos[top_z].to(dtype=dtype)  # (B, K, d)

    if only_type_inv:
        return _soft_project_type_inv_rel_to_targets(
            module, top_z, s_z, valid_z, target_ids, all_pos=all_pos, tau=tau
        )

    if only_type:
        return _soft_project_type_rel_to_targets(
            module, z_pos, s_z, valid_z, target_ids, tau=tau
        )

    if only_normal:
        return _soft_project_rel_to_targets(
            module, z_pos, s_z, valid_z, r_t, target_ids, all_pos=all_pos, tau=tau
        )

    special_rels = {type_rel} if type_rel is not None else set()
    if has_type_inv and type_inv_rel is not None:
        special_rels.add(type_inv_rel)
    has_normal = len(rel_set - special_rels) > 0

    cur: Optional[th.Tensor] = None
    if has_normal:
        safe_r = r_t
        if type_inv_rel is not None:
            safe_r = th.where(safe_r == type_inv_rel, th.zeros_like(safe_r), safe_r)
        if type_rel is not None:
            safe_r = th.where(safe_r == type_rel, th.zeros_like(safe_r), safe_r)
        cur = _soft_project_rel_to_targets(
            module, z_pos, s_z, valid_z, safe_r, target_ids, all_pos=all_pos, tau=tau
        )

    if has_type:
        s_type = _soft_project_type_rel_to_targets(
            module, z_pos, s_z, valid_z, target_ids, tau=tau
        )
        if cur is None:
            cur = s_type
        else:
            t_mask = (r_t == type_rel).unsqueeze(1)
            cur = th.where(t_mask, s_type, cur)

    if has_type_inv:
        s_inv_w = _soft_project_type_inv_rel_to_targets(
            module, top_z, s_z, valid_z, target_ids, all_pos=all_pos, tau=tau
        )
        if cur is None:
            cur = s_inv_w
        else:
            il_mask = (r_t == type_inv_rel).unsqueeze(1)
            cur = th.where(il_mask, s_inv_w, cur)

    assert cur is not None
    return cur


def _select_detached_witnesses(
    S_det: th.Tensor,
    k: int,
) -> Tuple[th.Tensor, th.Tensor]:
    """Select top-``k`` witness indices and validity mask from detached beam scores ``S_det`` ``(B, E)``."""
    n_elem = S_det.shape[1]
    k_keep = min(k, n_elem)
    top_z = th.argsort(S_det, dim=1, descending=True, stable=True)[:, :k_keep]  # (B, K)
    s_z_det = th.gather(S_det, 1, top_z)  # (B, K)
    valid_z = s_z_det > _EMPTY_DEPTH_FLOOR  # (B, K)
    k_eff = max(
        1,
        int(np.count_nonzero(valid_z.detach().cpu().numpy().any(axis=0))),
    )
    if k_eff < k_keep:
        top_z = top_z[:, :k_eff]
        valid_z = valid_z[:, :k_eff]
    return top_z, valid_z


def _beam_project_rel_topk(
    module: GroundedELModule,
    top_z: th.Tensor,
    s_z: th.Tensor,
    valid_z: th.Tensor,
    rel_ids: th.Tensor,
    *,
    k: int,
    chunk: int,
    all_pos: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Project witnesses ``(top_z, s_z, valid_z)`` along ``rel_ids`` and return exact top-``k`` ``(next_z, next_s, next_valid)``."""
    bsz, k_keep = top_z.shape
    n_elem, dim = all_pos.shape
    dtype = all_pos.dtype
    device = all_pos.device
    k_out = min(k, n_elem)

    if bsz == 0 or n_elem == 0 or k_out == 0:
        return (
            th.empty((bsz, k_out), dtype=th.long, device=device),
            th.empty((bsz, k_out), dtype=dtype, device=device),
            th.empty((bsz, k_out), dtype=th.bool, device=device),
        )

    rel_reg = module.translation_regions(rel_ids)
    r_cen = rel_reg.cen.to(dtype=dtype)
    r_w = rel_reg.w.to(dtype=dtype)
    r_act = rel_reg.active
    kr = r_cen.shape[1]

    k_eff = max(
        1,
        int(np.count_nonzero(valid_z.detach().cpu().numpy().any(axis=0))),
    )
    if k_eff < k_keep:
        top_z = top_z[:, :k_eff]
        s_z = s_z[:, :k_eff]
        valid_z = valid_z[:, :k_eff]
        k_keep = k_eff

    has_r_act = r_act.any(dim=1)
    has_valid_z = valid_z.any(dim=1)
    can_use_cdist = (
        not th.is_grad_enabled()
        and device.type == "cpu"
        and n_elem > 256
        and kr > 0
        and k_keep > 0
        and bool(has_r_act.detach().cpu().numpy().all())
        and bool(has_valid_z.detach().cpu().numpy().all())
    )
    if not can_use_cdist:
        S_in = th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)
        if k_keep > 0:
            s_masked = th.where(
                valid_z, s_z, th.full_like(s_z, _EMPTY_DEPTH_FLOOR)
            )
            S_in.scatter_(1, top_z, s_masked)
        S_out = _beam_project_rel(
            module, S_in, rel_ids, k=k_keep, chunk=chunk, all_pos=all_pos
        )
        next_z = th.argsort(S_out, dim=1, descending=True, stable=True)[:, :k_out]
        next_s = th.gather(S_out, 1, next_z)
        next_v = next_s > _EMPTY_DEPTH_FLOOR
        return next_z, next_s, next_v

    z_pos = all_pos[top_z].to(dtype=dtype)
    cen_zm = (z_pos.unsqueeze(2) + r_cen.unsqueeze(1)).contiguous()
    w_min_raw = r_w.amin(dim=-1)
    w_max_raw = r_w.amax(dim=-1)
    w_min = th.where(r_act, w_min_raw, th.full_like(w_min_raw, float("-inf")))
    dw = th.where(r_act, w_max_raw - w_min_raw, th.zeros_like(w_min_raw))

    stages: List[Tuple[int, int]] = []
    u_cur = 0
    for step in (1, 1, 2, 4):
        if u_cur < k_keep:
            u_next = min(k_keep, u_cur + step)
            stages.append((u_cur, u_next))
            u_cur = u_next
    if u_cur < k_keep:
        stages.append((u_cur, k_keep))

    c_b = 32
    out_z: List[th.Tensor] = []
    out_s: List[th.Tensor] = []
    out_v: List[th.Tensor] = []
    for b0 in range(0, bsz, c_b):
        nb = min(c_b, bsz - b0)
        cen_b = cen_zm[b0 : b0 + nb]
        sz_b = s_z[b0 : b0 + nb]
        vz_b = valid_z[b0 : b0 + nb]
        wmin_b = w_min[b0 : b0 + nb]
        dw_b = dw[b0 : b0 + nb]
        rw_b = r_w[b0 : b0 + nb]
        ract_b = r_act[b0 : b0 + nb]
        all_ract_b = bool(ract_b.detach().cpu().numpy().all())
        all_vz_b = bool(vz_b.detach().cpu().numpy().all())

        lb = th.full((nb, n_elem), float("-inf"), dtype=dtype, device=device)
        ub = th.full((nb, n_elem), float("-inf"), dtype=dtype, device=device)
        tau = th.full((nb, 1), float("-inf"), dtype=dtype, device=device)

        for u0, u1 in stages:
            act_bu = vz_b[:, u0:u1] & (sz_b[:, u0:u1] >= tau)
            b_idx, u_rel = th.nonzero(act_bu, as_tuple=True)
            if b_idx.numel() == 0:
                continue
            u_idx = u_rel + u0
            m_sel = b_idx.numel()
            cen_sel = cen_b[b_idx, u_idx]
            wmin_sel = wmin_b[b_idx]
            dw_sel = dw_b[b_idx]
            sz_sel = sz_b[b_idx, u_idx]

            d_inf = th.cdist(
                cen_sel.reshape(-1, dim), all_pos, p=float("inf")
            ).view(m_sel, kr, n_elem)
            d_inf.neg_()
            d_inf.add_(wmin_sel[:, :, None])
            lb_m = d_inf.amax(dim=1)
            th.minimum(lb_m, sz_sel[:, None], out=lb_m)

            d_inf.add_(dw_sel[:, :, None])
            ub_m = d_inf.amax(dim=1)
            th.minimum(ub_m, sz_sel[:, None], out=ub_m)

            idx_exp = b_idx.unsqueeze(1).expand(m_sel, n_elem)
            lb.scatter_reduce_(0, idx_exp, lb_m, reduce="amax", include_self=True)
            ub.scatter_reduce_(0, idx_exp, ub_m, reduce="amax", include_self=True)
            tau = th.topk(
                lb, k=k_out, dim=1, largest=True, sorted=False
            ).values.amin(dim=1, keepdim=True)

        ge_lb = lb >= tau
        can_tie = (ge_lb.cumsum(dim=1) - ge_lb.to(th.long)) < k_out
        surv_mask = (ub > tau) | ((ub == tau) & can_tie)
        n_surv = int(surv_mask.sum(dim=1).max().detach().cpu().numpy().flat[0])
        k_cand = min(n_elem, max(k_out, n_surv))
        ub_masked = th.where(surv_mask, ub, th.full_like(ub, float("-inf")))
        _, cand_idx = th.topk(ub_masked, k=k_cand, dim=1, largest=True, sorted=False)
        cand_idx, _ = th.sort(cand_idx, dim=1)

        c_cand = max(1, min(k_cand, (1 << 22) // max(1, nb * k_keep * kr * dim)))
        s_cand_chunks: List[th.Tensor] = []
        for m0 in range(0, k_cand, c_cand):
            c_sub = cand_idx[:, m0 : m0 + c_cand]
            p_cand = all_pos[c_sub].to(dtype=dtype)
            coord_dep = rw_b[:, None, None, :, :] - th.abs(
                p_cand[:, :, None, None, :] - cen_b[:, None, :, :, :]
            )
            slot_dep = coord_dep.amin(dim=-1)
            if not all_ract_b:
                slot_dep = th.where(
                    ract_b[:, None, None, :],
                    slot_dep,
                    th.full_like(slot_dep, float("-inf")),
                )
            dep_r = slot_dep.amax(dim=-1)
            cand_val = th.minimum(sz_b.unsqueeze(1), dep_r)
            if not all_vz_b:
                cand_val = th.where(
                    vz_b.unsqueeze(1),
                    cand_val,
                    th.full_like(cand_val, float("-inf")),
                )
            s_cand_chunks.append(cand_val.amax(dim=-1))
        s_cand = (
            s_cand_chunks[0]
            if len(s_cand_chunks) == 1
            else th.cat(s_cand_chunks, dim=1)
        )

        local_top = th.argsort(s_cand, dim=1, descending=True, stable=True)[:, :k_out]
        tz = th.gather(cand_idx, 1, local_top)
        sz = th.gather(s_cand, 1, local_top)
        vz = sz > _EMPTY_DEPTH_FLOOR
        out_z.append(tz)
        out_s.append(sz)
        out_v.append(vz)

    return th.cat(out_z, dim=0), th.cat(out_s, dim=0), th.cat(out_v, dim=0)


def _apply_beam_hop_topk(
    module: GroundedELModule,
    top_z: th.Tensor,
    s_z: th.Tensor,
    valid_z: th.Tensor,
    hop_rels_tuple: Tuple[int, ...],
    *,
    is_first_hop: bool,
    anchor_cls_ids: Optional[th.Tensor],
    all_anchors_are_classes: bool,
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    k: int,
    chunk: int,
    all_pos: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Apply a single relation hop to witnesses ``(top_z, s_z, valid_z)`` and return top-``k`` ``(next_z, next_s, next_valid)``."""
    bsz = top_z.shape[0]
    n_elem = all_pos.shape[0]
    device = all_pos.device
    dtype = all_pos.dtype
    k_out = min(k, n_elem)
    if bsz == 0 or n_elem == 0 or k_out == 0:
        return (
            th.empty((bsz, k_out), dtype=th.long, device=device),
            th.empty((bsz, k_out), dtype=dtype, device=device),
            th.empty((bsz, k_out), dtype=th.bool, device=device),
        )

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
    # ``type_inv_rel`` after the first hop: sources are the class elements among the witnesses
    has_type_inv_later = (
        (not is_first_hop)
        and type_inv_rel is not None
        and (type_inv_rel in rel_set)
    )
    only_type_inv_later = has_type_inv_later and rel_set == {type_inv_rel}
    only_normal = (not has_type_inv) and (not has_type) and (not has_type_inv_later)

    if only_normal:
        nz, ns, nv = _beam_project_rel_topk(
            module, top_z, s_z, valid_z, r_t, k=k_out, chunk=chunk, all_pos=all_pos
        )
        k_eff = max(
            1,
            int(np.count_nonzero(nv.detach().cpu().numpy().any(axis=0))),
        )
        if k_eff < k_out:
            nz, ns, nv = nz[:, :k_eff], ns[:, :k_eff], nv[:, :k_eff]
        return nz, ns, nv

    if only_type_inv:
        assert anchor_cls_ids is not None
        S_inv = score_entities(
            module, class_boxes(module, anchor_cls_ids), chunk=chunk
        )
        nz = th.argsort(S_inv, dim=1, descending=True, stable=True)[:, :k_out]
        ns = th.gather(S_inv, 1, nz)
        nv = ns > _EMPTY_DEPTH_FLOOR
        k_eff = max(
            1,
            int(np.count_nonzero(nv.detach().cpu().numpy().any(axis=0))),
        )
        if k_eff < k_out:
            nz, ns, nv = nz[:, :k_eff], ns[:, :k_eff], nv[:, :k_eff]
        return nz, ns, nv

    if only_type_inv_later:
        S_in_il = th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)
        if top_z.shape[1] > 0:
            S_in_il.scatter_(
                1,
                top_z,
                th.where(valid_z, s_z, th.full_like(s_z, _EMPTY_DEPTH_FLOOR)),
            )
        S_il = _beam_project_type_inv_rel(module, S_in_il, k=k, chunk=chunk)
        nz = th.argsort(S_il, dim=1, descending=True, stable=True)[:, :k_out]
        ns = th.gather(S_il, 1, nz)
        nv = ns > _EMPTY_DEPTH_FLOOR
        k_eff = max(
            1,
            int(np.count_nonzero(nv.detach().cpu().numpy().any(axis=0))),
        )
        if k_eff < k_out:
            nz, ns, nv = nz[:, :k_eff], ns[:, :k_eff], nv[:, :k_eff]
        return nz, ns, nv

    if only_type:
        S_in = th.full((bsz, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)
        if top_z.shape[1] > 0:
            S_in.scatter_(
                1,
                top_z,
                th.where(valid_z, s_z, th.full_like(s_z, _EMPTY_DEPTH_FLOOR)),
            )
        S_type = _beam_project_type_rel(module, S_in, k=k, all_pos=all_pos)
        nz = th.argsort(S_type, dim=1, descending=True, stable=True)[:, :k_out]
        ns = th.gather(S_type, 1, nz)
        nv = ns > _EMPTY_DEPTH_FLOOR
        k_eff = max(
            1,
            int(np.count_nonzero(nv.detach().cpu().numpy().any(axis=0))),
        )
        if k_eff < k_out:
            nz, ns, nv = nz[:, :k_eff], ns[:, :k_eff], nv[:, :k_eff]
        return nz, ns, nv

    # Partition batch rows into disjoint masks: inv_mask, type_mask, norm_mask
    if has_type_inv and anchor_cls_ids is not None:
        inv_mask = (r_t == type_inv_rel) & (anchor_cls_ids >= 0)
    else:
        inv_mask = th.zeros((bsz,), dtype=th.bool, device=device)

    if has_type:
        type_mask = (~inv_mask) & (r_t == type_rel)
    else:
        type_mask = th.zeros((bsz,), dtype=th.bool, device=device)

    if has_type_inv_later:
        il_mask = (~inv_mask) & (~type_mask) & (r_t == type_inv_rel)
    else:
        il_mask = th.zeros((bsz,), dtype=th.bool, device=device)

    special_rels = set()
    if has_type_inv and all_anchors_are_classes and type_inv_rel is not None:
        special_rels.add(type_inv_rel)
    if has_type_inv_later and type_inv_rel is not None:
        special_rels.add(type_inv_rel)
    if has_type and type_rel is not None:
        special_rels.add(type_rel)
    has_normal = len(rel_set - special_rels) > 0

    out_z = th.zeros((bsz, k_out), dtype=th.long, device=device)
    out_s = th.full((bsz, k_out), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device)
    out_v = th.zeros((bsz, k_out), dtype=th.bool, device=device)

    if has_normal:
        norm_mask = (~inv_mask) & (~type_mask) & (~il_mask)
        norm_idx = th.nonzero(norm_mask, as_tuple=True)[0]
        if norm_idx.numel() > 0:
            safe_r = r_t[norm_idx]
            if type_inv_rel is not None:
                safe_r = th.where(safe_r == type_inv_rel, th.zeros_like(safe_r), safe_r)
            if type_rel is not None:
                safe_r = th.where(safe_r == type_rel, th.zeros_like(safe_r), safe_r)
            nz_n, ns_n, nv_n = _beam_project_rel_topk(
                module,
                top_z[norm_idx],
                s_z[norm_idx],
                valid_z[norm_idx],
                safe_r,
                k=k_out,
                chunk=chunk,
                all_pos=all_pos,
            )
            out_z[norm_idx] = nz_n
            out_s[norm_idx] = ns_n
            out_v[norm_idx] = nv_n

    if has_type:
        type_idx = th.nonzero(type_mask, as_tuple=True)[0]
        if type_idx.numel() > 0:
            n_t = type_idx.numel()
            S_in_t = th.full(
                (n_t, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device
            )
            if top_z.shape[1] > 0:
                tz_t = top_z[type_idx]
                sz_t = s_z[type_idx]
                vz_t = valid_z[type_idx]
                S_in_t.scatter_(
                    1,
                    tz_t,
                    th.where(vz_t, sz_t, th.full_like(sz_t, _EMPTY_DEPTH_FLOOR)),
                )
            S_type = _beam_project_type_rel(module, S_in_t, k=k, all_pos=all_pos)
            nz_t = th.argsort(S_type, dim=1, descending=True, stable=True)[:, :k_out]
            ns_t = th.gather(S_type, 1, nz_t)
            nv_t = ns_t > _EMPTY_DEPTH_FLOOR
            out_z[type_idx] = nz_t
            out_s[type_idx] = ns_t
            out_v[type_idx] = nv_t

    if has_type_inv_later:
        il_idx = th.nonzero(il_mask, as_tuple=True)[0]
        if il_idx.numel() > 0:
            n_il = il_idx.numel()
            S_in_il = th.full(
                (n_il, n_elem), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device
            )
            if top_z.shape[1] > 0:
                sz_il = s_z[il_idx]
                S_in_il.scatter_(
                    1,
                    top_z[il_idx],
                    th.where(
                        valid_z[il_idx], sz_il, th.full_like(sz_il, _EMPTY_DEPTH_FLOOR)
                    ),
                )
            S_il = _beam_project_type_inv_rel(module, S_in_il, k=k, chunk=chunk)
            nz_il = th.argsort(S_il, dim=1, descending=True, stable=True)[:, :k_out]
            ns_il = th.gather(S_il, 1, nz_il)
            nv_il = ns_il > _EMPTY_DEPTH_FLOOR
            out_z[il_idx] = nz_il
            out_s[il_idx] = ns_il
            out_v[il_idx] = nv_il

    if has_type_inv:
        assert anchor_cls_ids is not None
        inv_idx = th.nonzero(inv_mask, as_tuple=True)[0]
        if inv_idx.numel() > 0:
            safe_cls = th.clamp_min(anchor_cls_ids[inv_idx], 0)
            S_inv = score_entities(
                module, class_boxes(module, safe_cls), chunk=chunk
            )
            nz_i = th.argsort(S_inv, dim=1, descending=True, stable=True)[:, :k_out]
            ns_i = th.gather(S_inv, 1, nz_i)
            nv_i = ns_i > _EMPTY_DEPTH_FLOOR
            out_z[inv_idx] = nz_i
            out_s[inv_idx] = ns_i
            out_v[inv_idx] = nv_i

    k_eff = max(
        1,
        int(np.count_nonzero(out_v.detach().cpu().numpy().any(axis=0))),
    )
    if k_eff < k_out:
        out_z = out_z[:, :k_eff]
        out_s = out_s[:, :k_eff]
        out_v = out_v[:, :k_eff]
    return out_z, out_s, out_v


def _score_witness_recursive(
    module: GroundedELModule,
    structure: Any,
    queries: Sequence[Any],
    target_ids: th.Tensor,
    *,
    e2el_t: Optional[th.Tensor],
    c2e_t: Optional[th.Tensor],
    c2e_arr: Optional[np.ndarray],
    type_rel: Optional[int],
    type_inv_rel: Optional[int],
    k: int,
    chunk: int,
    all_pos: th.Tensor,
    all_pos_det: th.Tensor,
    tau: th.Tensor,
) -> th.Tensor:
    """Recursively compute differentiable witness-chain soft scores of shape ``(B, M)`` at ``target_ids``."""
    bsz, m_targ = target_ids.shape
    n_elem = module.nb_elements
    device = all_pos.device
    dtype = all_pos.dtype

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
            safe_elem_ids = th.zeros_like(elem_ids)
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
            if safe_elem_ids.numel() > 0 and bool(
                ((safe_elem_ids < 0) | (safe_elem_ids >= n_elem)).any()
            ):
                raise ValueError(
                    f"Anchor element id out of range [0, {n_elem})"
                )

        if n_hops == 0:
            return th.where(
                target_ids == safe_elem_ids.unsqueeze(1),
                th.full((bsz, m_targ), _TOP_DEPTH_CEIL, dtype=dtype, device=device),
                th.full((bsz, m_targ), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device),
            )

        if n_hops == 1:
            return _apply_soft_hop_from_anchor(
                module,
                safe_elem_ids,
                hop_tuples[0],
                target_ids,
                anchor_cls_ids=anchor_cls_ids,
                all_anchors_are_classes=all_anchors_are_classes,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                all_pos=all_pos,
                tau=tau,
            )

        # Multi-hop path (n_hops >= 2): select top-k witnesses per intermediate variable under no_grad
        witness_sets: List[Tuple[th.Tensor, th.Tensor]] = []
        with th.no_grad():
            tz_det = safe_elem_ids.unsqueeze(1)
            if hop0_only_type_inv:
                sz_det = th.full(
                    (bsz, 1), _EMPTY_DEPTH_FLOOR, dtype=dtype, device=device
                )
                vz_det = th.zeros((bsz, 1), dtype=th.bool, device=device)
            else:
                sz_det = th.full(
                    (bsz, 1), _TOP_DEPTH_CEIL, dtype=dtype, device=device
                )
                vz_det = th.ones((bsz, 1), dtype=th.bool, device=device)
            for hop in range(n_hops - 1):
                tz_det, sz_det, vz_det = _apply_beam_hop_topk(
                    module,
                    tz_det,
                    sz_det,
                    vz_det,
                    hop_tuples[hop],
                    is_first_hop=(hop == 0),
                    anchor_cls_ids=anchor_cls_ids if hop == 0 else None,
                    all_anchors_are_classes=all_anchors_are_classes if hop == 0 else False,
                    type_rel=type_rel,
                    type_inv_rel=type_inv_rel,
                    k=k,
                    chunk=chunk,
                    all_pos=all_pos_det,
                )
                witness_sets.append((tz_det, vz_det))

        top_z, valid_z = witness_sets[0]
        s_z = _apply_soft_hop_from_anchor(
            module,
            safe_elem_ids,
            hop_tuples[0],
            top_z,
            anchor_cls_ids=anchor_cls_ids,
            all_anchors_are_classes=all_anchors_are_classes,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            all_pos=all_pos,
            tau=tau,
        )
        for hop in range(1, n_hops - 1):
            next_z, next_valid = witness_sets[hop]
            s_z = _apply_soft_hop_from_witnesses(
                module,
                top_z,
                s_z,
                valid_z,
                hop_tuples[hop],
                next_z,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                all_pos=all_pos,
                tau=tau,
            )
            top_z, valid_z = next_z, next_valid

        return _apply_soft_hop_from_witnesses(
            module,
            top_z,
            s_z,
            valid_z,
            hop_tuples[n_hops - 1],
            target_ids,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            all_pos=all_pos,
            tau=tau,
        )

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

        witness_sets_comp: List[Tuple[th.Tensor, th.Tensor]] = []
        with th.no_grad():
            S_det = _score_beam_recursive(
                module,
                sub_struct,
                sub_queries,
                e2el_t=e2el_t,
                c2e_t=c2e_t,
                c2e_arr=c2e_arr,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                k=k,
                chunk=chunk,
                all_pos=all_pos_det,
            )
            witness_sets_comp.append(_select_detached_witnesses(S_det, k))
            for hop in range(n_hops - 1):
                S_det = _apply_beam_hop(
                    module,
                    S_det,
                    hop_tuples[hop],
                    is_first_hop=False,
                    anchor_cls_ids=None,
                    all_anchors_are_classes=False,
                    type_rel=type_rel,
                    type_inv_rel=type_inv_rel,
                    k=k,
                    chunk=chunk,
                    all_pos=all_pos_det,
                )
                witness_sets_comp.append(_select_detached_witnesses(S_det, k))

        top_z, valid_z = witness_sets_comp[0]
        s_z = _score_witness_recursive(
            module,
            sub_struct,
            sub_queries,
            top_z,
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            k=k,
            chunk=chunk,
            all_pos=all_pos,
            all_pos_det=all_pos_det,
            tau=tau,
        )
        for hop in range(n_hops - 1):
            next_z, next_valid = witness_sets_comp[hop + 1]
            s_z = _apply_soft_hop_from_witnesses(
                module,
                top_z,
                s_z,
                valid_z,
                hop_tuples[hop],
                next_z,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                all_pos=all_pos,
                tau=tau,
            )
            top_z, valid_z = next_z, next_valid

        return _apply_soft_hop_from_witnesses(
            module,
            top_z,
            s_z,
            valid_z,
            hop_tuples[n_hops - 1],
            target_ids,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            all_pos=all_pos,
            tau=tau,
        )

    # Case 3: Union (sub_1, sub_2, ..., ('u',))
    if structure[-1] == ("u",):
        if len(structure) < 3:
            raise ValueError(
                f"Union structure must have at least 2 disjuncts before ('u',), got {structure!r}"
            )
        n_disj = len(structure) - 1
        if len(cols) < n_disj:
            raise ValueError(f"Queries do not match union structure {structure!r}")
        S_acc = _score_witness_recursive(
            module,
            structure[0],
            cols[0],
            target_ids,
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            k=k,
            chunk=chunk,
            all_pos=all_pos,
            all_pos_det=all_pos_det,
            tau=tau,
        )
        for i in range(1, n_disj):
            S_i = _score_witness_recursive(
                module,
                structure[i],
                cols[i],
                target_ids,
                e2el_t=e2el_t,
                c2e_t=c2e_t,
                c2e_arr=c2e_arr,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                k=k,
                chunk=chunk,
                all_pos=all_pos,
                all_pos_det=all_pos_det,
                tau=tau,
            )
            S_acc = th.maximum(S_acc, S_i)
        return S_acc

    # Case 4: Intersection (sub_1, sub_2, ...)
    n_conj = len(structure)
    if len(cols) != n_conj:
        raise ValueError(f"Queries do not match intersection structure {structure!r}")
    S_acc = _score_witness_recursive(
        module,
        structure[0],
        cols[0],
        target_ids,
        e2el_t=e2el_t,
        c2e_t=c2e_t,
        c2e_arr=c2e_arr,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        k=k,
        chunk=chunk,
        all_pos=all_pos,
        all_pos_det=all_pos_det,
        tau=tau,
    )
    for i in range(1, n_conj):
        S_i = _score_witness_recursive(
            module,
            structure[i],
            cols[i],
            target_ids,
            e2el_t=e2el_t,
            c2e_t=c2e_t,
            c2e_arr=c2e_arr,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            k=k,
            chunk=chunk,
            all_pos=all_pos,
            all_pos_det=all_pos_det,
            tau=tau,
        )
        S_acc = th.minimum(S_acc, S_i)
    return S_acc


def score_query_witness(
    module: GroundedELModule,
    structure: Any,
    queries: Sequence[Any],
    target_element_ids: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, th.Tensor],
    *,
    k: int = 16,
    type_rel: Optional[int] = None,
    type_inv_rel: Optional[int] = None,
    entity_to_element: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    class_of_entity: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    chunk: int = 65536,
) -> th.Tensor:
    """Compute differentiable witness-chain soft scores :math:`s(q, a)` at ``target_element_ids``.

    For each query :math:`q` in ``queries`` and each target element :math:`a` in
    ``target_element_ids``, selects the top-``k`` intermediate witness elements
    per intermediate variable using the detached crisp beam scores of
    :func:`score_query_beam`, and evaluates the maximum over those witness
    chains of the minimum over query atoms of the soft box depth at temperature
    :math:`\\tau = \\text{module.tau}`. Intersections take the minimum over
    branches and unions take the maximum over branches.

    Guarantees:
    - **Reasoner-free (R1):** Contains no imports or calls to reasoners or
      semantic completion routines.
    - **Zero-temperature equivalence with beam scoring:** As
      :math:`\\tau = \\text{module.tau} \\to 0^+`, ``score_query_witness``
      converges to the exact score of :func:`score_query_beam` gathered at
      ``target_element_ids`` for every supported structure (``1p``, ``2p``,
      ``3p``, ``2i``, ``3i``, ``2u``, ``ip``, ``pi``, ``up``).
    - **Target-restricted differentiability:** Gradients flow through the soft
      depths of the chosen witness chain without constructing ``(B, E)`` autograd
      graphs over all domain elements.

    Non-guarantees:
    - Witness selection uses detached crisp beam scores; gradients do not
      propagate through the discrete ``argsort`` witness selection step.
    - For ``type_inv_rel`` at a non-first hop, the sources are the class
      elements among the top-``k`` witnesses of the preceding variable, whereas
      :func:`score_query_beam` uses the top-``k`` class elements; the two agree
      at :math:`\\tau \\to 0^+` when the preceding hop is ``type_rel`` (only class
      elements are reachable) or ``k >= module.nb_elements``, but may differ otherwise.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module in ``"translation"`` role mode.
    structure : Any
        Query2Box structure tuple describing the query syntax tree.
    queries : Sequence[Any]
        Batch of query instances matching ``structure``.
    target_element_ids : Union[int, Sequence[int], Sequence[Sequence[int]], numpy.ndarray, torch.Tensor]
        Target element indices in ``[0, module.nb_elements)`` of shape ``(B,)``
        or ``(B, M)``.
    k : int, default 16
        Number of top witness elements retained per intermediate variable :math:`\\ge 1`.
    type_rel : Optional[int], default None
        Optional relation id representing ``rdf:type`` (concept-level step).
    type_inv_rel : Optional[int], default None
        Optional relation id representing ``rdf:type^-`` (class-instance step).
    entity_to_element : Optional[Union[Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to module element ids in
        ``[0, module.nb_elements)``.
    class_of_entity : Optional[Union[Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to class ids in
        ``[0, module.nb_classes)`` (or ``-1`` for non-class entities).
    chunk : int, default 65536
        Element chunk size :math:`\\ge 1` for detached beam witness selection.

    Returns
    -------
    torch.Tensor
        Soft witness scores :math:`s(q, a)` of shape ``(B,)`` when
        ``target_element_ids`` is 1-D or ``(B, M)`` when 2-D.
    """
    _validate_module(module, require_translation=True)
    k_val = _validate_int_ge("k", k, 1)
    chunk_sz = _validate_int_ge("chunk", chunk, 1)
    if not isinstance(queries, (list, tuple)):
        raise TypeError(
            f"queries must be a Sequence (list or tuple) of query instances, got {type(queries).__name__}"
        )

    bsz = len(queries)
    device = module.elem_pos.weight.device
    dtype = module.elem_pos.weight.dtype
    n_elem = module.nb_elements

    targ_t, was_1d = _to_2d_long_targets(
        target_element_ids,
        bsz,
        n_elem,
        device,
        "target_element_ids",
        allow_empty_cols=True,
    )
    if bsz == 0 or targ_t.shape[1] == 0:
        return (
            th.empty((bsz,), dtype=dtype, device=device)
            if was_1d
            else th.empty((bsz, targ_t.shape[1]), dtype=dtype, device=device)
        )

    e2el_t: Optional[th.Tensor] = None
    if entity_to_element is not None:
        e2el_t = _to_1d_long(entity_to_element, device, "entity_to_element")

    c2e_t: Optional[th.Tensor] = None
    c2e_arr: Optional[np.ndarray] = None
    if class_of_entity is not None:
        c2e_t = _to_1d_long(class_of_entity, device, "class_of_entity")
        c2e_arr = c2e_t.detach().cpu().numpy()

    all_pos = module.positions()  # (E, d)
    all_pos_det = all_pos.detach()
    scores = _score_witness_recursive(
        module,
        structure,
        queries,
        targ_t,
        e2el_t=e2el_t,
        c2e_t=c2e_t,
        c2e_arr=c2e_arr,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        k=k_val,
        chunk=chunk_sz,
        all_pos=all_pos,
        all_pos_det=all_pos_det,
        tau=module.tau,
    )
    return scores.squeeze(1) if was_1d else scores


def query_witness_loss(
    module: GroundedELModule,
    structure: Any,
    queries: Sequence[Any],
    answers: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, th.Tensor],
    negatives: Optional[
        Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, th.Tensor]
    ] = None,
    k: int = 16,
    margin: float = 0.0,
    type_rel: Optional[int] = None,
    type_inv_rel: Optional[int] = None,
    entity_to_element: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    class_of_entity: Optional[Union[Sequence[int], np.ndarray, th.Tensor]] = None,
    chunk: int = 65536,
    return_scores: bool = False,
) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]]:
    """Compute the differentiable witness max-min margin loss for multi-hop queries.

    For each query :math:`q` in ``queries``, each positive answer element
    :math:`a` in ``answers``, and each negative answer element :math:`a^-` in
    ``negatives``, evaluates the witness-chain soft score

    .. math::

        s(q, y) = \\max_{(z_1, \\dots, z_m) \\in Z_1 \\times \\dots \\times Z_m}
                  \\min_{\\alpha \\in \\text{atoms}(q)} \\operatorname{soft\\_depth}_\\tau(\\alpha; z, y),

    where the top-``k`` witness sets :math:`Z_i` per intermediate variable are
    selected by the detached crisp beam scores of :func:`score_query_beam` and
    each atom's soft depth aggregates over active box slots using temperature
    :math:`\\tau = \\text{module.tau}` (:math:`m^* + \\tau \\log \\sum_m \\exp((d_m - m^*)/\\tau)`).
    Intersections take the minimum over branches and unions take the maximum
    over branches. Returns the margin loss

    .. math::

        \\mathcal{L} = \\operatorname{mean}\\big(\\operatorname{relu}(\\text{margin} - s(q, a))\\big)
                    + \\operatorname{mean}\\big(\\operatorname{relu}(\\text{margin} + s(q, a^-))\\big)

    (with the negative term omitted when ``negatives is None`` or empty).

    Guarantees:
    - **Reasoner-free (R1):** Contains no imports or calls to reasoners or
      semantic completion routines.
    - **Beam-scorer consistency:** As :math:`\\tau = \\text{module.tau} \\to 0^+`,
      the witness score :math:`s(q, a)` equals :func:`score_query_beam`'s score
      for the same query and answer element on ``1p``, ``2p``, ``3p``, ``2i``,
      ``3i``, ``2u``, ``ip``, ``pi``, and ``up`` queries.
    - **Differentiable chain supervision:** Gradients flow through the soft
      depths of the winning witness chain and active box components without
      calling ``.item()`` or building ``(B, E)`` autograd graphs.

    Non-guarantees:
    - Top-``k`` intermediate witness selection is performed under
      :func:`torch.no_grad`; gradients do not flow through the discrete
      ``argsort`` selection of :math:`Z_i`.
    - When all positive scores satisfy :math:`s(q, a) > \\text{margin}` and all
      negative scores satisfy :math:`s(q, a^-) < -\\text{margin}`, the ReLU
      hinge is flat at ``0.0`` with zero gradient.

    Parameters
    ----------
    module : GroundedELModule
        Grounded EL module in ``"translation"`` role mode.
    structure : Any
        Query2Box structure tuple describing the query syntax tree (e.g.
        ``2p``, ``3p``, ``2i``, ``3i``).
    queries : Sequence[Any]
        Batch of query instances matching ``structure`` of length ``B``.
    answers : Union[int, Sequence[int], Sequence[Sequence[int]], numpy.ndarray, torch.Tensor]
        Positive answer element indices in ``[0, module.nb_elements)`` of shape
        ``(B,)`` or ``(B, P)`` with ``P >= 1``.
    negatives : Optional[Union[int, Sequence[int], Sequence[Sequence[int]], numpy.ndarray, torch.Tensor]], default None
        Optional negative answer element indices in ``[0, module.nb_elements)``
        of shape ``(B,)`` or ``(B, N_neg)``.
    k : int, default 16
        Number of top witness elements retained per intermediate variable :math:`\\ge 1`.
    margin : float, default 0.0
        Finite hinge margin for positive and negative scores.
    type_rel : Optional[int], default None
        Optional relation id representing ``rdf:type`` (concept-level step).
    type_inv_rel : Optional[int], default None
        Optional relation id representing ``rdf:type^-`` (class-instance step).
    entity_to_element : Optional[Union[Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to module element ids in
        ``[0, module.nb_elements)``.
    class_of_entity : Optional[Union[Sequence[int], numpy.ndarray, torch.Tensor]], default None
        Optional 1-D mapping from dataset entity ids to class ids in
        ``[0, module.nb_classes)`` (or ``-1`` for non-class entities).
    chunk : int, default 65536
        Element chunk size :math:`\\ge 1` for detached beam witness selection.
    return_scores : bool, default False
        If ``True``, returns ``(loss, pos_scores, neg_scores)`` instead of
        ``loss`` alone.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
        Scalar loss tensor of shape ``()``, or ``(loss, pos_scores, neg_scores)``
        when ``return_scores=True``.
    """
    _validate_module(module, require_translation=True)
    k_val = _validate_int_ge("k", k, 1)
    margin_f = _validate_finite_margin("margin", margin)
    chunk_sz = _validate_int_ge("chunk", chunk, 1)
    if not isinstance(return_scores, bool):
        raise TypeError(
            f"return_scores must be a bool, got {type(return_scores).__name__}"
        )
    if not isinstance(queries, (list, tuple)):
        raise TypeError(
            f"queries must be a Sequence (list or tuple) of query instances, got {type(queries).__name__}"
        )

    bsz = len(queries)
    device = module.elem_pos.weight.device
    dtype = module.elem_pos.weight.dtype
    n_elem = module.nb_elements

    ans_t, ans_was_1d = _to_2d_long_targets(
        answers, bsz, n_elem, device, "answers", allow_empty_cols=False
    )
    n_pos = ans_t.shape[1]

    neg_t: Optional[th.Tensor] = None
    neg_was_1d = False
    if negatives is not None:
        neg_t, neg_was_1d = _to_2d_long_targets(
            negatives, bsz, n_elem, device, "negatives", allow_empty_cols=True
        )

    if bsz == 0:
        zero_loss = module.elem_pos.weight.sum() * 0.0
        if not return_scores:
            return zero_loss
        empty_pos = (
            th.empty((0,), dtype=dtype, device=device)
            if ans_was_1d
            else th.empty((0, n_pos), dtype=dtype, device=device)
        )
        empty_neg: Optional[th.Tensor] = None
        if neg_t is not None:
            empty_neg = (
                th.empty((0,), dtype=dtype, device=device)
                if neg_was_1d
                else th.empty((0, neg_t.shape[1]), dtype=dtype, device=device)
            )
        return zero_loss, empty_pos, empty_neg

    e2el_t: Optional[th.Tensor] = None
    if entity_to_element is not None:
        e2el_t = _to_1d_long(entity_to_element, device, "entity_to_element")

    c2e_t: Optional[th.Tensor] = None
    c2e_arr: Optional[np.ndarray] = None
    if class_of_entity is not None:
        c2e_t = _to_1d_long(class_of_entity, device, "class_of_entity")
        c2e_arr = c2e_t.detach().cpu().numpy()

    if neg_t is not None and neg_t.shape[1] > 0:
        targets = th.cat([ans_t, neg_t], dim=1)
    else:
        targets = ans_t

    all_pos = module.positions()  # (E, d)
    all_pos_det = all_pos.detach()
    all_scores = _score_witness_recursive(
        module,
        structure,
        queries,
        targets,
        e2el_t=e2el_t,
        c2e_t=c2e_t,
        c2e_arr=c2e_arr,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        k=k_val,
        chunk=chunk_sz,
        all_pos=all_pos,
        all_pos_det=all_pos_det,
        tau=module.tau,
    )

    pos_2d = all_scores[:, :n_pos]
    loss = th.relu(margin_f - pos_2d).mean()

    neg_out: Optional[th.Tensor] = None
    if neg_t is not None:
        if neg_t.shape[1] > 0:
            neg_2d = all_scores[:, n_pos:]
            loss = loss + th.relu(margin_f + neg_2d).mean()
            neg_out = neg_2d.squeeze(1) if neg_was_1d else neg_2d
        else:
            neg_out = th.empty((bsz, 0), dtype=dtype, device=device)

    if not return_scores:
        return loss
    pos_out = pos_2d.squeeze(1) if ans_was_1d else pos_2d
    return loss, pos_out, neg_out

