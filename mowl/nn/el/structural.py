"""Region algebra with signed half-widths, structural GCI losses, definition tables, and existence targets.

Provides the building blocks for the ``objective="structural"`` formulation of
:class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.PrismEL`:

1. **Regions with absent components (:class:`Regions`):** Each class is
   materialised as :math:`K` component slots with centre :math:`c \\in \\mathbb{R}^d`,
   *signed* half-width :math:`w \\in \\mathbb{R}^d` (negative in a coordinate iff the
   slot is empty along that coordinate), and a boolean ``active`` mask.
2. **Signed overlap and non-emptiness (:func:`signed_overlap`, :func:`nonempty_measure`,
   :func:`region_size`, :func:`pair_size`):** Differentiable signed measures that
   remain informative when component pairs are separated, routing gradient to the
   nearest active component or pair.
3. **Structural GCI and existence losses (:func:`inclusion_tight`,
   :func:`gci0_structural`, :func:`gci0_bot_structural`, :func:`gci1_structural`,
   :func:`gci1_bot_structural`, :func:`existence_loss`):** Squared hinge losses with
   detached non-emptiness gates and single-coordinate separation for
   :math:`C \\sqcap D \\sqsubseteq \\bot`.
4. **Definition table (:class:`DefinitionTable`) and existence targets
   (:class:`ExistenceTargets`):** Compile acyclic conjunction definitions
   :math:`A \\equiv X \\sqcap Y` within the slot/depth budget and identify
   satisfiable classes and conjunction pairs requiring positive measure.

Guarantees:
- **Zero-loss faithfulness at margin 0:** With ``enabled=True`` on
  :class:`DefinitionTable`, at ``margin = 0``, the sum of
  :func:`gci0_structural`, :func:`gci0_bot_structural`, :func:`gci1_structural`,
  :func:`gci1_bot_structural`, and :func:`existence_loss` is ``0`` if and only if
  the materialised regions satisfy all box normal forms set-theoretically and
  every target class and target conjunction pair has a non-empty intersection
  (up to boundary contact :math:`w_m = 0`).
- **Single-coordinate disjointness in :func:`gci1_bot_structural`:** Two active
  component boxes separated by at least ``margin`` in *at least one* coordinate
  incur zero loss, even when their projections overlap in all remaining
  coordinates.
- **Vacuous satisfaction in :func:`gci1_structural`:** When all active component
  pairs of :math:`C` and :math:`D` have :math:`\\text{ne}(C_i \\cap D_j) \\le 0`
  (or when the row is masked by :class:`DefinitionTable`), :func:`gci1_structural`
  returns exact ``0.0`` with zero gradient.

Non-guarantees:
- **Hard max routing in :func:`region_size` and :func:`pair_size`:** Gradients of
  :func:`region_size` and :func:`pair_size` flow only to the argmax active slot
  or pair (ties broken by PyTorch's ``amax``); spare dead components require
  discrete re-seeding to move when trapped in local minima.
- **Boundary contact at ``margin = 0``:** At ``margin = 0``, boxes with zero width
  (:math:`\\text{ne} = 0`) sit on the boundary between empty and non-empty; a
  strictly positive ``margin > 0`` during training is required to enforce
  strictly positive interior volume and strict separation.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Set, Tuple

import torch as th

if TYPE_CHECKING:
    from mowl.geometric.completion import CanonicalModel

__all__ = [
    "Regions",
    "signed_overlap",
    "pair_slots",
    "nonempty_measure",
    "region_size",
    "pair_size",
    "inclusion_tight",
    "gci0_structural",
    "gci0_bot_structural",
    "gci1_structural",
    "gci1_bot_structural",
    "existence_loss",
    "DefinitionTable",
    "ExistenceTargets",
]


class Regions(NamedTuple):
    """Materialised multi-component box regions with signed half-widths and active mask.

    Attributes
    ----------
    cen : torch.Tensor
        Component centres of shape ``(B, K, d)``.
    w : torch.Tensor
        Signed half-widths of shape ``(B, K, d)``; negative in coordinate ``m``
        if and only if the box is empty along coordinate ``m``.
    active : torch.Tensor
        Boolean mask of shape ``(B, K)`` indicating which slots are active.
    """

    cen: th.Tensor
    w: th.Tensor
    active: th.Tensor


def _pair_bounds(
    cen_a: th.Tensor,
    w_a: th.Tensor,
    cen_b: th.Tensor,
    w_b: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Compute pairwise ``(lower, upper, cen_ij, w_ij)`` of shape ``(B, ka, kb, d)``."""
    ca = cen_a.unsqueeze(2)
    wa = w_a.unsqueeze(2)
    cb = cen_b.unsqueeze(1)
    wb = w_b.unsqueeze(1)
    lower = th.maximum(ca - wa, cb - wb)
    upper = th.minimum(ca + wa, cb + wb)
    cen_ij = (lower + upper) / 2.0
    w_ij = (upper - lower) / 2.0
    return lower, upper, cen_ij, w_ij


def signed_overlap(
    cen_a: th.Tensor,
    w_a: th.Tensor,
    cen_b: th.Tensor,
    w_b: th.Tensor,
) -> th.Tensor:
    """Compute coordinatewise signed overlap between two sets of component boxes.

    For each pair :math:`(a, b)` and coordinate :math:`m`, evaluates

    .. math::

        \\text{sov}_m(a, b)
        = \\min(c_{a,m} + w_{a,m},\\, c_{b,m} + w_{b,m})
        - \\max(c_{a,m} - w_{a,m},\\, c_{b,m} - w_{b,m}).

    Guarantees:
    - Returns a tensor of shape ``(B, ka, kb, d)`` with the same dtype and device
      as the inputs.
    - Positive in coordinate ``m`` when the intervals overlap (equal to the
      intersection length), zero when they touch at a point, and negative when
      separated (equal to the negative gap).

    Non-guarantees:
    - Does not apply ``torch.abs`` to ``w_a`` or ``w_b``; callers must pass
      signed half-widths (``|offset|`` for primitive boxes).

    Parameters
    ----------
    cen_a : torch.Tensor
        Centres of the first region set, shape ``(B, ka, d)``.
    w_a : torch.Tensor
        Signed half-widths of the first region set, shape ``(B, ka, d)``.
    cen_b : torch.Tensor
        Centres of the second region set, shape ``(B, kb, d)``.
    w_b : torch.Tensor
        Signed half-widths of the second region set, shape ``(B, kb, d)``.

    Returns
    -------
    torch.Tensor
        Pairwise coordinatewise signed overlaps of shape ``(B, ka, kb, d)``.
    """
    lower, upper, _, _ = _pair_bounds(cen_a, w_a, cen_b, w_b)
    return upper - lower


def pair_slots(a: Regions, b: Regions, max_slots: int) -> Regions:
    """Materialise pairwise intersection boxes of two :class:`Regions` into ``max_slots`` slots.

    All :math:`k_a \\times k_b` pairs :math:`(i, j)` are formed with
    :math:`\\text{lower} = \\max(c_a - w_a, c_b - w_b)`,
    :math:`\\text{upper} = \\min(c_a + w_a, c_b + w_b)`,
    :math:`c = (\\text{lower} + \\text{upper}) / 2`, and
    :math:`w = (\\text{upper} - \\text{lower}) / 2`. A pair is active iff both parent
    slots are active. Slots in each batch row are stably reordered so that all
    active pairs come first, then truncated or padded to ``max_slots`` with
    inactive padding ``cen = 0`` and ``w = -1``.

    Guarantees:
    - Output :class:`Regions` has shapes ``(B, max_slots, d)`` for ``cen`` and
      ``w`` and ``(B, max_slots)`` for ``active``.
    - Active pairs appear at the leading positions of each row in their original
      lexicographic ``(i, j)`` order.
    - All inactive output slots have ``cen = 0`` and ``w = -1`` in every coordinate.
    - Raises :class:`ValueError` if any batch row has more than ``max_slots``
      active pairs or if ``max_slots < 1``.

    Non-guarantees:
    - Active pair slots may still represent geometrically empty intersections
      (indicated by ``w[..., m] < 0`` in one or more coordinates).

    Parameters
    ----------
    a : Regions
        First parent regions of shape ``(B, ka, d)``.
    b : Regions
        Second parent regions of shape ``(B, kb, d)``.
    max_slots : int
        Target number of component slots :math:`K \\ge 1`.

    Returns
    -------
    Regions
        Pairwise intersection regions of shape ``(B, max_slots, d)``.
    """
    if max_slots < 1:
        raise ValueError(f"max_slots must be >= 1, got {max_slots}")

    bsz, ka, dim = a.cen.shape
    kb = b.cen.shape[1]
    total_pairs = ka * kb

    _, _, cen_grid, w_grid = _pair_bounds(a.cen, a.w, b.cen, b.w)
    cen_ij = cen_grid.reshape(bsz, total_pairs, dim)
    w_ij = w_grid.reshape(bsz, total_pairs, dim)
    act_ij = (a.active.unsqueeze(2) & b.active.unsqueeze(1)).reshape(bsz, total_pairs)

    active_counts = act_ij.sum(dim=1)
    if bsz > 0 and bool((active_counts > max_slots).any().item()):
        max_act = int(active_counts.max().item())
        raise ValueError(
            f"pair_slots produced {max_act} active pairs in a row, exceeding max_slots={max_slots}."
        )

    # Stable sort placing active pairs (True) before inactive pairs (False)
    idx_grid = th.arange(total_pairs, device=act_ij.device).unsqueeze(0)
    sort_key = th.where(act_ij, idx_grid, idx_grid + total_pairs)
    order = th.argsort(sort_key, dim=1)
    order_d = order.unsqueeze(-1).expand(bsz, total_pairs, dim)

    cen_sorted = th.gather(cen_ij, 1, order_d)
    w_sorted = th.gather(w_ij, 1, order_d)
    act_sorted = th.gather(act_ij, 1, order)

    if total_pairs >= max_slots:
        cen_out = cen_sorted[:, :max_slots, :]
        w_out = w_sorted[:, :max_slots, :]
        act_out = act_sorted[:, :max_slots]
    else:
        pad_n = max_slots - total_pairs
        pad_cen = th.zeros((bsz, pad_n, dim), dtype=a.cen.dtype, device=a.cen.device)
        pad_w = -th.ones((bsz, pad_n, dim), dtype=a.w.dtype, device=a.w.device)
        pad_act = th.zeros((bsz, pad_n), dtype=th.bool, device=a.active.device)
        cen_out = th.cat([cen_sorted, pad_cen], dim=1)
        w_out = th.cat([w_sorted, pad_w], dim=1)
        act_out = th.cat([act_sorted, pad_act], dim=1)

    act_mask_d = act_out.unsqueeze(-1)
    cen_out = th.where(act_mask_d, cen_out, th.zeros_like(cen_out))
    w_out = th.where(act_mask_d, w_out, -th.ones_like(w_out))
    return Regions(cen=cen_out, w=w_out, active=act_out)


def nonempty_measure(w: th.Tensor) -> th.Tensor:
    """Compute the minimum signed side length :math:`\\text{ne}(a) = \\min_m 2 w_m` across coordinates.

    Guarantees:
    - Reduces the last dimension ``d`` of ``w`` (e.g. ``(B, K, d) -> (B, K)``).
    - Strictly positive if and only if every coordinate has :math:`w_m > 0`.

    Non-guarantees:
    - Gradients with respect to ``w`` flow only to the coordinate attaining the
      minimum side length.

    Parameters
    ----------
    w : torch.Tensor
        Signed half-widths of shape ``(..., d)``.

    Returns
    -------
    torch.Tensor
        Minimum signed widths ``2 * min_m w_m`` of shape ``(...)``.
    """
    return 2.0 * th.amin(w, dim=-1)


def region_size(r: Regions) -> th.Tensor:
    """Compute :math:`\\text{NE}(C) = \\max_{i \\text{ active}} \\text{ne}(C_i)` for each row in ``r``.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Inactive slots are masked with ``-inf`` before the maximum and never
      contribute values or gradients.
    - Raises :class:`ValueError` if any row in ``r`` has zero active slots.

    Non-guarantees:
    - Gradients flow only to the active slot attaining the maximum.

    Parameters
    ----------
    r : Regions
        Batch of regions of shape ``(B, K, d)``.

    Returns
    -------
    torch.Tensor
        Per-row non-emptiness measure of shape ``(B,)``.
    """
    if r.active.ndim != 2:
        raise ValueError(f"r.active must be 2-D (B, K), got shape {tuple(r.active.shape)}")
    if r.active.shape[0] > 0 and not bool(r.active.any(dim=1).all().item()):
        raise ValueError("region_size requires at least one active slot per batch row.")
    ne = nonempty_measure(r.w)
    masked_ne = ne.masked_fill(~r.active, float("-inf"))
    return th.amax(masked_ne, dim=1)


def pair_size(a: Regions, b: Regions) -> th.Tensor:
    """Compute :math:`\\text{NE}(C \\sqcap D) = \\max_{(i,j) \\text{ active}} \\min_m \\text{sov}_m(C_i, D_j)`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Inactive pairs are masked with ``-inf`` before the maximum; when all active
      pairs are disjoint, selects the least separated active pair so gradient
      pulls that pair together.
    - Raises :class:`ValueError` if any batch row has zero active pairs.

    Non-guarantees:
    - Gradients flow only to the single active pair attaining the maximum.

    Parameters
    ----------
    a : Regions
        First regions of shape ``(B, ka, d)``.
    b : Regions
        Second regions of shape ``(B, kb, d)``.

    Returns
    -------
    torch.Tensor
        Per-row pairwise non-emptiness measure of shape ``(B,)``.
    """
    bsz = a.cen.shape[0]
    act_ij = (a.active.unsqueeze(2) & b.active.unsqueeze(1)).reshape(bsz, -1)
    if bsz > 0 and not bool(act_ij.any(dim=1).all().item()):
        raise ValueError("pair_size requires at least one active component pair per batch row.")
    sov = signed_overlap(a.cen, a.w, b.cen, b.w)
    ne_ij = th.amin(sov, dim=-1).reshape(bsz, -1)
    masked_ne = ne_ij.masked_fill(~act_ij, float("-inf"))
    return th.amax(masked_ne, dim=1)


def _inclusion_tight_violation(
    cen_a: th.Tensor,
    w_a: th.Tensor,
    cen_b: th.Tensor,
    w_b: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Compute coordinatewise tight inclusion violations ``relu(|c_a - c_b| + w_a - w_b + m)`` of shape ``(B, ka, kb, d)``."""
    ca = cen_a.unsqueeze(2)
    wa = w_a.unsqueeze(2)
    cb = cen_b.unsqueeze(1)
    wb = w_b.unsqueeze(1)
    return th.relu(th.abs(ca - cb) + wa - wb + margin)


def inclusion_tight(
    cen_a: th.Tensor,
    w_a: th.Tensor,
    cen_b: th.Tensor,
    w_b: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Compute tight pairwise inclusion scores :math:`\\|\\text{relu}(|c_a - c_b| + w_a - w_b + m)\\|_2`.

    Uses the tightening margin convention (:math:`m \\ge 0` tightens containment).

    Guarantees:
    - Returns a tensor of shape ``(B, ka, kb)``.
    - At ``margin = 0``, entry ``(b, i, j)`` is ``0`` if and only if
      :math:`[c_{a,i} - w_{a,i},\\, c_{a,i} + w_{a,i}] \\subseteq [c_{b,j} - w_{b,j},\\, c_{b,j} + w_{b,j}]`
      in every coordinate (for non-empty :math:`a_i`).

    Non-guarantees:
    - For an empty source slot with sufficiently negative ``w_a``, the hinge can
      be zero even when ``cen_a`` lies outside ``b``; callers gate source slots
      by :math:`\\text{ne} > 0`.

    Parameters
    ----------
    cen_a : torch.Tensor
        Source centres of shape ``(B, ka, d)``.
    w_a : torch.Tensor
        Source signed half-widths of shape ``(B, ka, d)``.
    cen_b : torch.Tensor
        Target centres of shape ``(B, kb, d)``.
    w_b : torch.Tensor
        Target signed half-widths of shape ``(B, kb, d)``.
    margin : float
        Tightening inclusion margin :math:`m`.

    Returns
    -------
    torch.Tensor
        Pairwise inclusion distances of shape ``(B, ka, kb)``.
    """
    violation = _inclusion_tight_violation(cen_a, w_a, cen_b, w_b, margin)
    return th.linalg.norm(violation, dim=-1)


def gci0_structural(c: Regions, d: Regions, margin: float) -> th.Tensor:
    """Compute the structural GCI0 loss for :math:`C \\sqsubseteq D`.

    Evaluates

    .. math::

        \\text{mean}_{i:\\, \\text{active},\\, \\text{ne}(C_i) > 0}^{\\text{det}}
        \\min_{j \\text{ active}} \\text{incl}_m(C_i, D_j)^2,

    returning ``0`` for rows with no active non-empty :math:`C_i`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Non-emptiness gate :math:`\\text{ne}(C_i) > 0` is detached so inclusion
      violations never push :math:`C_i` to shrink to empty.
    - At ``margin = 0``, equals ``0`` iff every active non-empty slot of :math:`C`
      is contained in some active slot of :math:`D`.
    - Raises :class:`ValueError` if any batch row of ``d`` has zero active slots.

    Non-guarantees:
    - Vacuously ``0`` when :math:`C` has no active non-empty slots; non-emptiness
      of satisfiable classes is enforced separately by :func:`existence_loss`.

    Parameters
    ----------
    c : Regions
        Sub-concept regions of shape ``(B, kc, d)``.
    d : Regions
        Super-concept regions of shape ``(B, kd, d)``.
    margin : float
        Tightening inclusion margin.

    Returns
    -------
    torch.Tensor
        Per-row squared GCI0 structural loss of shape ``(B,)``.
    """
    if d.active.shape[0] > 0 and not bool(d.active.any(dim=1).all().item()):
        raise ValueError("gci0_structural requires at least one active target slot in d per batch row.")
    incl_sq = _inclusion_tight_violation(c.cen, c.w, d.cen, d.w, margin).square().sum(dim=-1)
    incl_masked = incl_sq.masked_fill(~d.active.unsqueeze(1), float("inf"))
    min_j = th.amin(incl_masked, dim=2)  # (B, kc)

    ne_c_det = nonempty_measure(c.w.detach())
    gate_bool = c.active & (ne_c_det > 0.0)
    gate = gate_bool.to(dtype=c.cen.dtype)

    num = th.where(gate_bool, min_j, th.zeros_like(min_j)).sum(dim=1)
    den = gate.sum(dim=1).clamp_min(1.0)
    return num / den


def gci0_bot_structural(c: Regions) -> th.Tensor:
    """Compute the structural GCI0-bot loss :math:`\\text{mean}_{i \\text{ active}} \\text{relu}(\\text{ne}(C_i))^2` for :math:`C \\sqsubseteq \\bot`.

    Uses no margin because primitive half-widths satisfy :math:`w = |o| \\ge 0`,
    so a positive margin would impose a non-zero floor on primitive classes.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Equals ``0`` iff every active slot of :math:`C` has :math:`\\text{ne}(C_i) \\le 0`.

    Non-guarantees:
    - Only penalises the minimum coordinate width of each active slot.

    Parameters
    ----------
    c : Regions
        Concept regions of shape ``(B, kc, d)``.

    Returns
    -------
    torch.Tensor
        Per-row squared GCI0-bot structural loss of shape ``(B,)``.
    """
    ne_c = nonempty_measure(c.w)
    term = th.relu(ne_c).square()
    mask = c.active.to(dtype=c.cen.dtype)
    num = th.where(c.active, term, th.zeros_like(term)).sum(dim=1)
    den = mask.sum(dim=1).clamp_min(1.0)
    return num / den


def gci1_structural(
    c: Regions,
    d: Regions,
    e: Regions,
    margin: float,
    masked: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the structural GCI1 loss for :math:`C \\sqcap D \\sqsubseteq E`.

    Evaluates

    .. math::

        \\text{mean}_{(i,j):\\, \\text{active},\\, \\text{ne}(C_i \\cap D_j) > 0}^{\\text{det}}
        \\min_{l \\text{ active}} \\text{incl}_m(I_{ij}, E_l)^2,

    where :math:`I_{ij}` is the pair slot for :math:`(C_i, D_j)`. Returns exact
    ``0`` when no active pair is non-empty (vacuous) or when the row is marked
    in ``masked``.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Rows with ``masked[b] == True`` or with all active pairs satisfying
      :math:`\\text{ne}(C_i \\cap D_j) \\le 0` evaluate to exact ``0.0`` with zero
      gradient.
    - Imposes no emptiness penalty on :math:`C \\cap D` (conjunction non-emptiness
      belongs exclusively to :func:`existence_loss`).
    - Raises :class:`ValueError` if any batch row of ``e`` has zero active slots.

    Non-guarantees:
    - Requires each non-empty pair box :math:`C_i \\cap D_j` to be covered by a
      single active slot :math:`E_l` rather than a union of multiple slots of :math:`E`.

    Parameters
    ----------
    c : Regions
        First conjunct regions of shape ``(B, kc, d)``.
    d : Regions
        Second conjunct regions of shape ``(B, kd, d)``.
    e : Regions
        Super-concept regions of shape ``(B, ke, d)``.
    margin : float
        Tightening inclusion margin.
    masked : Optional[torch.Tensor], default None
        Boolean tensor of shape ``(B,)`` marking definition rows whose loss is ``0``.

    Returns
    -------
    torch.Tensor
        Per-row squared GCI1 structural loss of shape ``(B,)``.
    """
    if e.active.shape[0] > 0 and not bool(e.active.any(dim=1).all().item()):
        raise ValueError("gci1_structural requires at least one active target slot in e per batch row.")

    bsz, kc, dim = c.cen.shape
    kd = d.cen.shape[1]
    total_pairs = kc * kd

    _, _, cen_grid, w_grid = _pair_bounds(c.cen, c.w, d.cen, d.w)
    inter_cen = cen_grid.reshape(bsz, total_pairs, dim)
    inter_w = w_grid.reshape(bsz, total_pairs, dim)
    pair_active = (c.active.unsqueeze(2) & d.active.unsqueeze(1)).reshape(bsz, total_pairs)

    ne_ij_det = nonempty_measure(inter_w.detach())
    gate_bool = pair_active & (ne_ij_det > 0.0)
    if masked is not None:
        mask_bool = masked.to(dtype=th.bool)
        gate_bool = gate_bool & (~mask_bool.unsqueeze(1))

    incl_sq = _inclusion_tight_violation(inter_cen, inter_w, e.cen, e.w, margin).square().sum(dim=-1)
    incl_masked = incl_sq.masked_fill(~e.active.unsqueeze(1), float("inf"))
    min_l = th.amin(incl_masked, dim=2)  # (B, kc * kd)

    gate = gate_bool.to(dtype=c.cen.dtype)
    num = th.where(gate_bool, min_l, th.zeros_like(min_l)).sum(dim=1)
    den = gate.sum(dim=1).clamp_min(1.0)
    out = num / den
    if masked is not None:
        out = th.where(masked.to(dtype=th.bool), th.zeros_like(out), out)
    return out


def gci1_bot_structural(c: Regions, d: Regions, margin: float) -> th.Tensor:
    """Compute the structural GCI1-bot loss :math:`\\text{mean}_{(i,j) \\text{ active}} \\text{relu}(\\text{ne}(C_i \\cap D_j) + m)^2` for :math:`C \\sqcap D \\sqsubseteq \\bot`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Equals ``0`` if and only if every active component pair :math:`(C_i, D_j)`
      is separated by at least ``margin`` in *at least one* coordinate
      (:math:`\\min_m \\text{sov}_m(C_i, D_j) \\le -m`).

    Non-guarantees:
    - Gradients for each violating pair flow only through the coordinate with the
      smallest signed overlap :math:`\\arg\\min_m \\text{sov}_m(C_i, D_j)`.

    Parameters
    ----------
    c : Regions
        First conjunct regions of shape ``(B, kc, d)``.
    d : Regions
        Second conjunct regions of shape ``(B, kd, d)``.
    margin : float
        Separation margin :math:`m`.

    Returns
    -------
    torch.Tensor
        Per-row squared GCI1-bot structural loss of shape ``(B,)``.
    """
    bsz = c.cen.shape[0]
    sov = signed_overlap(c.cen, c.w, d.cen, d.w)
    ne_ij = th.amin(sov, dim=-1).reshape(bsz, -1)
    pair_active = (c.active.unsqueeze(2) & d.active.unsqueeze(1)).reshape(bsz, -1)

    term = th.relu(ne_ij + margin).square()
    mask = pair_active.to(dtype=c.cen.dtype)
    num = th.where(pair_active, term, th.zeros_like(term)).sum(dim=1)
    den = mask.sum(dim=1).clamp_min(1.0)
    return num / den


def existence_loss(
    class_sizes: th.Tensor,
    pair_sizes: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Compute the structural existence loss over satisfiable classes and conjunction pairs.

    Evaluates

    .. math::

        \\text{mean}_{C \\in \\text{Sat}} \\text{relu}(m - \\text{NE}(C))^2
        + \\text{mean}_{(C, D) \\in \\text{SatPairs}} \\text{relu}(m - \\text{NE}(C \\sqcap D))^2,

    where an empty ``class_sizes`` or ``pair_sizes`` tensor contributes ``0.0``.

    Guarantees:
    - Returns a scalar ``torch.Tensor`` of shape ``()``.
    - Equals ``0`` iff every entry in ``class_sizes`` and ``pair_sizes`` is at
      least ``margin``.

    Non-guarantees:
    - Does not deduplicate targets; callers should pass deduplicated target sizes
      from :class:`ExistenceTargets`.

    Parameters
    ----------
    class_sizes : torch.Tensor
        1-D tensor of shape ``(C,)`` holding :math:`\\text{NE}(C)` for target classes.
    pair_sizes : torch.Tensor
        1-D tensor of shape ``(P,)`` holding :math:`\\text{NE}(C \\sqcap D)` for target pairs.
    margin : float
        Minimum target non-emptiness measure :math:`m`.

    Returns
    -------
    torch.Tensor
        Scalar existence loss.
    """
    dtype = class_sizes.dtype if class_sizes.numel() > 0 else pair_sizes.dtype
    device = class_sizes.device if class_sizes.numel() > 0 else pair_sizes.device

    if class_sizes.numel() > 0:
        cls_term = th.relu(margin - class_sizes).square().mean()
    else:
        cls_term = th.zeros((), dtype=dtype, device=device)

    if pair_sizes.numel() > 0:
        pair_term = th.relu(margin - pair_sizes).square().mean()
    else:
        pair_term = th.zeros((), dtype=dtype, device=device)

    return cls_term + pair_term


@dataclass
class DefinitionTable:
    """Compiled structural conjunction definitions :math:`A := X \\sqcap Y` extracted from normal forms.

    Guarantees:
    - The accepted definition graph is a DAG: ``level[a] == 1 + max(level[def_x[a]], level[def_y[a]])``
      for every defined class ``a``, and ``level[p] == 0`` for every primitive class ``p``.
    - For every defined class ``a``, ``comps[a] == comps[def_x[a]] * comps[def_y[a]] <= max_components``
      and ``level[a] <= max_level`` (when a bound was passed to :meth:`from_normal_forms`).
    - :meth:`masked_gci0` marks exactly the rows ``(A, X)`` and ``(A, Y)`` of
      accepted definitions :math:`A \\to (X, Y)`, and :meth:`masked_gci1` marks
      ``(X, Y, A)`` and ``(Y, X, A)``.

    Non-guarantees:
    - Only compiles binary conjunction definitions witnessed simultaneously by
      ``(A, X)``, ``(A, Y)`` in ``gci0`` and ``(X, Y, A)`` in ``gci1``; role
      definitions (:math:`A \\equiv \\exists r.B`) remain primitive.

    Attributes
    ----------
    def_mask : torch.Tensor
        Boolean tensor of shape ``(n_classes,)`` indicating accepted defined classes.
    def_x : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with first definer index (``-1`` for primitive).
    def_y : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with second definer index (``-1`` for primitive).
    level : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with definition depth (``0`` for primitive).
    comps : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with required slot count (``k`` for primitive).
    n_defined : int
        Number of accepted defined classes.
    max_level : int
        Maximum definition level among all classes (``0`` if none defined).
    """

    def_mask: th.Tensor
    def_x: th.Tensor
    def_y: th.Tensor
    level: th.Tensor
    comps: th.Tensor
    n_defined: int
    max_level: int
    _masked_gci0_keys: th.Tensor
    _masked_gci1_keys: th.Tensor

    @property
    def n_classes(self) -> int:
        """Total number of classes in the table."""
        return int(self.def_mask.shape[0])

    @classmethod
    def from_normal_forms(
        cls,
        gci0: th.Tensor,
        gci1: th.Tensor,
        n_classes: int,
        k: int,
        max_components: int,
        top_idx: Optional[int],
        bottom_idx: Optional[int],
        enabled: bool = True,
        max_level: Optional[int] = None,
    ) -> DefinitionTable:
        """Build a :class:`DefinitionTable` from ``gci0`` and ``gci1`` normal-form index tensors.

        Guarantees:
        - When ``enabled=False``, every class is primitive (``n_defined == 0``,
          ``max_level == 0``, ``comps == k``) and no rows are masked.
        - Candidate definers for ``A`` are the lexicographically smallest
          ``(min(X, Y), max(X, Y))`` with ``X != Y``, ``A not in {X, Y}``,
          none of ``A, X, Y`` in ``{top_idx, bottom_idx}``, ``(A, X)`` and
          ``(A, Y)`` in ``gci0``, and ``(X, Y, A)`` in ``gci1``.
        - Candidates are accepted iff acyclic, ``comps(X) * comps(Y) <= max_components``,
          and ``level(A) <= max_level`` (when ``max_level is not None``), breaking
          dependency cycles deterministically in increasing class index.

        Non-guarantees:
        - Rejected candidates remain primitive and their ``gci0``/``gci1`` rows
          are not masked.

        Parameters
        ----------
        gci0 : torch.Tensor
            Long tensor of shape ``(N0, 2)`` with rows ``(C, D)``.
        gci1 : torch.Tensor
            Long tensor of shape ``(N1, 3)`` with rows ``(C, D, E)``.
        n_classes : int
            Total number of classes.
        k : int
            Number of primitive components per class.
        max_components : int
            Maximum allowed component slots :math:`K` for a defined class.
        top_idx : Optional[int]
            Class index of ``owl:Thing`` if present, else ``None``.
        bottom_idx : Optional[int]
            Class index of ``owl:Nothing`` if present, else ``None``.
        enabled : bool, default True
            If ``False``, disables structural definitions (all classes primitive).
        max_level : Optional[int], default None
            Optional maximum definition level cap (``None`` for unbounded).

        Returns
        -------
        DefinitionTable
            Compiled definition table.
        """
        def_mask = th.zeros(n_classes, dtype=th.bool)
        def_x = th.full((n_classes,), -1, dtype=th.long)
        def_y = th.full((n_classes,), -1, dtype=th.long)
        level = th.zeros(n_classes, dtype=th.long)
        comps = th.full((n_classes,), k, dtype=th.long)

        if not enabled or n_classes == 0 or (max_level is not None and max_level < 1):
            return cls(
                def_mask=def_mask,
                def_x=def_x,
                def_y=def_y,
                level=level,
                comps=comps,
                n_defined=0,
                max_level=0,
                _masked_gci0_keys=th.zeros(0, dtype=th.long),
                _masked_gci1_keys=th.zeros(0, dtype=th.long),
            )

        excluded: Set[int] = {
            int(idx) for idx in (top_idx, bottom_idx) if idx is not None
        }

        gci0_set: Set[Tuple[int, int]] = set()
        if gci0 is not None and gci0.numel() > 0:
            for row in gci0.detach().cpu().reshape(-1, 2).tolist():
                gci0_set.add((int(row[0]), int(row[1])))

        candidates: Dict[int, Tuple[int, int]] = {}
        if gci1 is not None and gci1.numel() > 0:
            for row in gci1.detach().cpu().reshape(-1, 3).tolist():
                x_raw, y_raw, a_raw = int(row[0]), int(row[1]), int(row[2])
                if x_raw == y_raw or a_raw == x_raw or a_raw == y_raw:
                    continue
                if a_raw in excluded or x_raw in excluded or y_raw in excluded:
                    continue
                if not (
                    0 <= a_raw < n_classes
                    and 0 <= x_raw < n_classes
                    and 0 <= y_raw < n_classes
                ):
                    continue
                if (a_raw, x_raw) in gci0_set and (a_raw, y_raw) in gci0_set:
                    pair = (min(x_raw, y_raw), max(x_raw, y_raw))
                    if a_raw not in candidates or pair < candidates[a_raw]:
                        candidates[a_raw] = pair

        accepted: Dict[int, Tuple[int, int]] = {}
        lvl_arr: List[int] = [0] * n_classes
        cmp_arr: List[int] = [k] * n_classes

        dependants: Dict[int, List[int]] = {u: [] for u in candidates}
        in_deg: Dict[int, int] = {}
        ready_heap: List[int] = []
        for u, (x, y) in candidates.items():
            deg = 0
            if x in candidates:
                dependants[x].append(u)
                deg += 1
            if y in candidates:
                dependants[y].append(u)
                deg += 1
            in_deg[u] = deg
            if deg == 0:
                heapq.heappush(ready_heap, u)

        undecided: Set[int] = set(candidates.keys())
        sorted_cands = sorted(candidates.keys())
        cand_ptr = 0

        def _release_dependants(node: int) -> None:
            for dep in dependants[node]:
                if dep in undecided:
                    in_deg[dep] -= 1
                    if in_deg[dep] == 0:
                        heapq.heappush(ready_heap, dep)

        while undecided:
            while ready_heap:
                u = heapq.heappop(ready_heap)
                if u not in undecided:
                    continue
                undecided.remove(u)
                x, y = candidates[u]
                lv = 1 + max(lvl_arr[x], lvl_arr[y])
                cp = cmp_arr[x] * cmp_arr[y]
                if cp <= max_components and (max_level is None or lv <= max_level):
                    accepted[u] = (x, y)
                    lvl_arr[u] = lv
                    cmp_arr[u] = cp
                _release_dependants(u)

            if undecided:
                while sorted_cands[cand_ptr] not in undecided:
                    cand_ptr += 1
                victim = sorted_cands[cand_ptr]
                cand_ptr += 1
                undecided.remove(victim)
                _release_dependants(victim)

        gci0_keys: List[int] = []
        gci1_keys: List[int] = []
        for a, (x, y) in sorted(accepted.items()):
            def_mask[a] = True
            def_x[a] = x
            def_y[a] = y
            level[a] = lvl_arr[a]
            comps[a] = cmp_arr[a]
            gci0_keys.append(a * n_classes + x)
            gci0_keys.append(a * n_classes + y)
            gci1_keys.append((x * n_classes + y) * n_classes + a)
            gci1_keys.append((y * n_classes + x) * n_classes + a)

        n_defined = len(accepted)
        max_lvl = int(level.max().item()) if n_defined > 0 else 0
        return cls(
            def_mask=def_mask,
            def_x=def_x,
            def_y=def_y,
            level=level,
            comps=comps,
            n_defined=n_defined,
            max_level=max_lvl,
            _masked_gci0_keys=th.tensor(gci0_keys, dtype=th.long),
            _masked_gci1_keys=th.tensor(gci1_keys, dtype=th.long),
        )

    def masked_gci0(self, rows: th.Tensor) -> th.Tensor:
        """Return a boolean mask over ``gci0`` rows ``(C, D)`` covered by accepted definitions.

        Guarantees:
        - Returns a 1-D ``torch.bool`` tensor of length ``rows.shape[0]`` on
          ``rows.device``.
        - Marks entry ``b`` as ``True`` iff ``rows[b]`` is ``(A, X)`` or
          ``(A, Y)`` for an accepted definition :math:`A \\to (X, Y)`.

        Non-guarantees:
        - Assumes row entries are valid class indices in ``0 .. n_classes - 1``.

        Parameters
        ----------
        rows : torch.Tensor
            Integer tensor of shape ``(B, 2)``.

        Returns
        -------
        torch.Tensor
            Boolean mask tensor of shape ``(B,)``.
        """
        bsz = rows.shape[0]
        if bsz == 0 or self._masked_gci0_keys.numel() == 0:
            return th.zeros(bsz, dtype=th.bool, device=rows.device)
        n = self.n_classes
        keys = rows[:, 0].to(th.long) * n + rows[:, 1].to(th.long)
        return th.isin(keys, self._masked_gci0_keys.to(device=rows.device))

    def masked_gci1(self, rows: th.Tensor) -> th.Tensor:
        """Return a boolean mask over ``gci1`` rows ``(C, D, E)`` covered by accepted definitions.

        Guarantees:
        - Returns a 1-D ``torch.bool`` tensor of length ``rows.shape[0]`` on
          ``rows.device``.
        - Marks entry ``b`` as ``True`` iff ``rows[b]`` is ``(X, Y, A)`` or
          ``(Y, X, A)`` for an accepted definition :math:`A \\to (X, Y)`.

        Non-guarantees:
        - Assumes row entries are valid class indices in ``0 .. n_classes - 1``.

        Parameters
        ----------
        rows : torch.Tensor
            Integer tensor of shape ``(B, 3)``.

        Returns
        -------
        torch.Tensor
            Boolean mask tensor of shape ``(B,)``.
        """
        bsz = rows.shape[0]
        if bsz == 0 or self._masked_gci1_keys.numel() == 0:
            return th.zeros(bsz, dtype=th.bool, device=rows.device)
        n = self.n_classes
        r0 = rows[:, 0].to(th.long)
        r1 = rows[:, 1].to(th.long)
        r2 = rows[:, 2].to(th.long)
        keys = (r0 * n + r1) * n + r2
        return th.isin(keys, self._masked_gci1_keys.to(device=rows.device))


@dataclass
class ExistenceTargets:
    """Satisfiable classes and conjunction pairs targeted by :func:`existence_loss`.

    Guarantees:
    - ``classes`` is a 1-D ``torch.long`` tensor of shape ``(C,)`` and ``pairs``
      is a 2-D ``torch.long`` tensor of shape ``(P, 2)`` with ``pairs[:, 0] < pairs[:, 1]``.
    - ``top_idx`` and ``bottom_idx`` are excluded from both ``classes`` and ``pairs``.
    - Pairs are unordered, deduplicated, and strictly satisfy ``C != D``.

    Non-guarantees:
    - Without a :class:`~mowl.geometric.completion.CanonicalModel`, satisfiability
      is approximated syntactically from ``gci0_bot`` and ``gci1_bot`` rows only.

    Attributes
    ----------
    classes : torch.Tensor
        1-D ``torch.long`` tensor of satisfiable class indices, shape ``(C,)``.
    pairs : torch.Tensor
        2-D ``torch.long`` tensor of satisfiable conjunction pairs, shape ``(P, 2)``.
    """

    classes: th.Tensor
    pairs: th.Tensor

    @classmethod
    def from_normal_forms(
        cls,
        gci0_bot: th.Tensor,
        gci1: th.Tensor,
        gci1_bot: th.Tensor,
        n_classes: int,
        top_idx: Optional[int],
        bottom_idx: Optional[int],
        index_to_iri: Optional[Dict[int, str]] = None,
        cm: Optional[CanonicalModel] = None,
    ) -> ExistenceTargets:
        """Build :class:`ExistenceTargets` from normal forms and an optional :class:`CanonicalModel`.

        Guarantees:
        - With ``cm`` provided, uses ``cm.is_satisfiable(iri)`` for classes whose
          IRI is in ``cm.concept_names`` (treating unknown IRIs as satisfiable)
          and ``cm.joint_satisfiable([iri_c, iri_d])`` for ``gci1`` pairs whose
          IRIs are both in ``cm.concept_names`` (falling back to the asserted
          ``gci0_bot``/``gci1_bot`` rule when an IRI is unknown to ``cm``).
          Raises :class:`ValueError` if ``cm`` is provided with ``index_to_iri=None``.
        - Without ``cm``, includes every class not appearing in ``gci0_bot`` and
          every distinct unordered pair ``(min(C, D), max(C, D))`` from ``gci1``
          where neither ``C`` nor ``D`` appears in ``gci0_bot`` and ``(C, D)``
          does not appear in either order in ``gci1_bot``.
        - Excludes ``top_idx`` and ``bottom_idx`` everywhere.

        Non-guarantees:
        - Does not test pairs that do not occur as the left-hand side of any
          ``gci1`` row.

        Parameters
        ----------
        gci0_bot : torch.Tensor
            Long tensor of shape ``(N, 2)`` with rows ``(C, bottom_idx)``.
        gci1 : torch.Tensor
            Long tensor of shape ``(N1, 3)`` with rows ``(C, D, E)``.
        gci1_bot : torch.Tensor
            Long tensor of shape ``(N2, 2)`` with rows ``(C, D)``.
        n_classes : int
            Total number of classes.
        top_idx : Optional[int]
            Class index of ``owl:Thing`` if present, else ``None``.
        bottom_idx : Optional[int]
            Class index of ``owl:Nothing`` if present, else ``None``.
        index_to_iri : Optional[Dict[int, str]], default None
            Mapping from class index to IRI string (required when ``cm`` is given).
        cm : Optional[CanonicalModel], default None
            Saturated canonical model for semantic satisfiability queries.

        Returns
        -------
        ExistenceTargets
            Target class and pair index tensors.
        """
        if cm is not None and index_to_iri is None:
            raise ValueError("index_to_iri must be provided when cm is not None.")

        excluded: Set[int] = {
            int(idx) for idx in (top_idx, bottom_idx) if idx is not None
        }

        unsat_classes_asserted: Set[int] = set()
        if gci0_bot is not None and gci0_bot.numel() > 0:
            for row in gci0_bot.detach().cpu().reshape(-1, 2).tolist():
                unsat_classes_asserted.add(int(row[0]))

        unsat_pairs_asserted: Set[Tuple[int, int]] = set()
        if gci1_bot is not None and gci1_bot.numel() > 0:
            for row in gci1_bot.detach().cpu().reshape(-1, 2).tolist():
                c_raw, d_raw = int(row[0]), int(row[1])
                unsat_pairs_asserted.add((min(c_raw, d_raw), max(c_raw, d_raw)))

        cm_concepts: Set[str] = set()
        if cm is not None:
            cn_attr = getattr(cm, "concept_names", set())
            cm_concepts = set(cn_attr() if callable(cn_attr) else cn_attr)

        target_classes: List[int] = []
        for c in range(n_classes):
            if c in excluded:
                continue
            if cm is not None:
                assert index_to_iri is not None
                iri = index_to_iri.get(c)
                if iri is not None and iri in cm_concepts:
                    if cm.is_satisfiable(iri):
                        target_classes.append(c)
                else:
                    target_classes.append(c)
            else:
                if c not in unsat_classes_asserted:
                    target_classes.append(c)

        candidate_pairs: Set[Tuple[int, int]] = set()
        if gci1 is not None and gci1.numel() > 0:
            for row in gci1.detach().cpu().reshape(-1, 3).tolist():
                c_raw, d_raw = int(row[0]), int(row[1])
                if c_raw == d_raw or c_raw in excluded or d_raw in excluded:
                    continue
                if not (0 <= c_raw < n_classes and 0 <= d_raw < n_classes):
                    continue
                candidate_pairs.add((min(c_raw, d_raw), max(c_raw, d_raw)))

        target_pairs: List[Tuple[int, int]] = []
        for c, d in sorted(candidate_pairs):
            if cm is not None:
                assert index_to_iri is not None
                iri_c = index_to_iri.get(c)
                iri_d = index_to_iri.get(d)
                if (
                    iri_c is not None
                    and iri_d is not None
                    and iri_c in cm_concepts
                    and iri_d in cm_concepts
                ):
                    if cm.joint_satisfiable([iri_c, iri_d]):
                        target_pairs.append((c, d))
                else:
                    if (
                        c not in unsat_classes_asserted
                        and d not in unsat_classes_asserted
                        and (c, d) not in unsat_pairs_asserted
                    ):
                        target_pairs.append((c, d))
            else:
                if (
                    c not in unsat_classes_asserted
                    and d not in unsat_classes_asserted
                    and (c, d) not in unsat_pairs_asserted
                ):
                    target_pairs.append((c, d))

        classes_tensor = th.tensor(target_classes, dtype=th.long)
        if target_pairs:
            pairs_tensor = th.tensor(target_pairs, dtype=th.long)
        else:
            pairs_tensor = th.zeros((0, 2), dtype=th.long)

        return cls(classes=classes_tensor, pairs=pairs_tensor)
