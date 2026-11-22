"""The structural region algebra and losses of MultiBoxEL and PrismEL: the direct
implementation (a frozen copy; the library computes them from the parts in
:mod:`mowl.representation.library.structural_reading` and
:mod:`mowl.representation.library.structural_regions`).

The definition tables, existence targets and witness targets are data, not losses: they are
in :mod:`mowl.nn.el.structural`, which both implementations read.

Provides the building blocks for the ``objective="structural"`` formulation of
:class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.PrismEL`:

1. **Regions with absent components (:class:`Regions`):** Each class is
   materialised as :math:`K` component slots with centre :math:`c \\in \\mathbb{R}^d`,
   *signed* half-width :math:`w \\in \\mathbb{R}^d` (negative in a coordinate iff the
   slot is empty along that coordinate), and a boolean ``active`` mask.
2. **Signed overlap, non-emptiness, and point-inside measures (:func:`signed_overlap`,
   :func:`nonempty_measure`, :func:`region_size`, :func:`pair_size`,
   :func:`inside_measure`):** Differentiable signed measures that remain
   informative when component pairs are separated or a witness point lies
   outside a region, routing gradient to the nearest active component or pair.
3. **Structural GCI, existence, and witness losses (:func:`inclusion_tight`,
   :func:`gci0_structural`, :func:`gci0_bot_structural`, :func:`gci1_structural`,
   :func:`gci1_bot_structural`, :func:`existence_loss`, :func:`witness_loss`):**
   Squared hinge losses with detached non-emptiness gates, single-coordinate
   separation for :math:`C \\sqcap D \\sqsubseteq \\bot`, and per-class witness
   supervision.
4. **Definition table (:class:`DefinitionTable`, :func:`slot_components`), existence targets
   (:class:`ExistenceTargets`), and witness targets (:class:`WitnessTargets`):**
   Compile acyclic conjunction definitions :math:`A \\equiv X \\sqcap Y` within
   the slot/depth budget, map materialised definition slots back to primitive
   component indices, identify satisfiable classes and conjunction pairs
   requiring positive measure, and build positive/negative class-membership
   targets for per-class witness points.

Guarantees:

- **Zero-loss faithfulness at margin 0:** With ``enabled=True`` on
  :class:`DefinitionTable`, at ``margin = 0``, zero sum of
  :func:`gci0_structural`, :func:`gci0_bot_structural`, :func:`gci1_structural`,
  :func:`gci1_bot_structural`, and :func:`existence_loss` implies that the
  materialised regions form a geometric model of the box normal forms in which
  every target class and target conjunction pair has a non-empty intersection
  (up to boundary contact :math:`w_m = 0`); the converse holds when every
  active non-empty source slot (in :func:`gci0_structural`) or pair slot (in
  :func:`gci1_structural`) is covered by a single active target slot (see the
  Non-guarantees below).
- **Single-coordinate disjointness in :func:`gci1_bot_structural`:** Two active
  component boxes separated by at least ``margin`` in *at least one* coordinate
  incur zero loss, even when their projections overlap in all remaining
  coordinates.
- **Vacuous satisfaction in :func:`gci1_structural`:** When all active component
  pairs of :math:`C` and :math:`D` have :math:`\\text{ne}(C_i \\cap D_j) \\le 0`
  (or when the row is masked by :class:`DefinitionTable`), :func:`gci1_structural`
  returns exact ``0.0`` with zero gradient.
- **Witness-loss zero set in :func:`witness_loss`:** At ``margin = m``,
  :func:`witness_loss` equals ``0`` if and only if for every positive pair
  :math:`(i, D) \\in \\text{pos}` the witness point :math:`p_i` lies at least
  :math:`m` inside some active slot of :math:`D`
  (:math:`\\text{inside}(p_i, D) \\ge m`) and for every negative pair
  :math:`(i, D) \\in \\text{neg}` the point :math:`p_i` lies at least :math:`m`
  outside every active slot of :math:`D`
  (:math:`\\text{inside}(p_i, D) \\le -m`).

Non-guarantees:

- **Hard max routing in :func:`region_size`, :func:`pair_size`, and :func:`inside_measure`:**
  Gradients of :func:`region_size`, :func:`pair_size`, and
  :func:`inside_measure` flow only to the argmax active slot or pair (ties
  broken by PyTorch's ``amax``); spare dead or idle components require discrete
  re-seeding to move when trapped in local minima. Rows with zero active slots
  in :func:`inside_measure` return the finite floor ``-1e6`` with zero gradient.
- **Boundary contact at ``margin = 0``:** At ``margin = 0``, boxes with zero width
  (:math:`\\text{ne} = 0`) sit on the boundary between empty and non-empty; a
  strictly positive ``margin > 0`` during training is required to enforce
  strictly positive interior volume and strict separation.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch as th

__all__ = [
    "Regions", "signed_overlap", "pair_slots", "nonempty_measure", "region_size", "pair_size",
    "inside_measure", "inclusion_tight", "gci0_structural", "gci0_bot_structural",
    "gci1_structural", "gci1_bot_structural", "existence_loss", "witness_loss",
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


def inside_measure(points: th.Tensor, regions: Regions) -> th.Tensor:
    """Compute signed interior depth :math:`\\max_{s \\text{ active}} \\min_m (w_{s,m} - |p_m - \\text{cen}_{s,m}|)` of ``points`` in ``regions``.

    Guarantees:

    - Returns a 1-D tensor of shape ``(W,)`` matching the batch dimension of ``points``.
    - Entry ``b`` is strictly positive (at least ``m``) if and only if ``points[b]``
      lies at least ``m`` inside some active slot of ``regions[b]`` in every coordinate.
    - Inactive slots are masked before the maximum; if a batch row has no active
      slots, its output is the finite constant ``-1e6`` with zero gradient so
      that downstream losses keep finite gradients.
    - Raises :class:`ValueError` if ``points`` and ``regions`` have incompatible shapes.

    Non-guarantees:

    - Gradients flow only to the active slot attaining the maximum and the
      coordinate attaining the minimum (zero gradient on rows with no active slots).

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates of shape ``(W, d)``.
    regions : Regions
        Target regions of shape ``(W, K, d)`` for ``cen`` and ``w`` and
        ``(W, K)`` for ``active``.

    Returns
    -------
    torch.Tensor
        Signed inside measure of shape ``(W,)``.
    """
    if points.ndim != 2:
        raise ValueError(
            f"points must be 2-D (W, d), got shape {tuple(points.shape)}"
        )
    if (
        regions.cen.ndim != 3
        or regions.w.ndim != 3
        or regions.active.ndim != 2
        or regions.cen.shape != regions.w.shape
        or regions.cen.shape[:2] != regions.active.shape
        or points.shape[0] != regions.cen.shape[0]
        or points.shape[1] != regions.cen.shape[2]
    ):
        raise ValueError(
            f"Incompatible shapes between points {tuple(points.shape)} and regions "
            f"(cen={tuple(regions.cen.shape)}, w={tuple(regions.w.shape)}, "
            f"active={tuple(regions.active.shape)})."
        )

    bsz = int(points.shape[0])
    if bsz == 0:
        return th.empty((0,), dtype=points.dtype, device=points.device)

    coord_depth = regions.w - th.abs(points.unsqueeze(1) - regions.cen)
    slot_depth = th.amin(coord_depth, dim=-1)  # (W, K)

    has_active = regions.active.any(dim=1, keepdim=True)  # (W, 1)
    inactive_fill = th.where(
        has_active,
        th.full_like(slot_depth, float("-inf")),
        th.full_like(slot_depth, -1e6),
    )
    masked = th.where(regions.active, slot_depth, inactive_fill)
    return th.amax(masked, dim=1)


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


def gci1_bot_structural(
    c: Regions,
    d: Regions,
    margin: float,
    agg: str = "mean",
) -> th.Tensor:
    """Compute the structural GCI1-bot loss for :math:`C \\sqcap D \\sqsubseteq \\bot`.

    Evaluates

    .. math::

        \\operatorname{agg}_{(i,j) \\text{ active}} \\operatorname{relu}(\\text{ne}(C_i \\cap D_j) + m)^2,

    where ``agg`` is ``"mean"`` (per-row average over active pairs, ``0`` when no
    pair is active) or ``"sum"`` (per-row sum over active pairs).

    Guarantees:

    - Returns a 1-D tensor of shape ``(B,)``.
    - Under both ``agg="mean"`` and ``agg="sum"``, equals ``0`` if and only if
      every active component pair :math:`(C_i, D_j)` is separated by at least
      ``margin`` in *at least one* coordinate
      (:math:`\\min_m \\text{sov}_m(C_i, D_j) \\le -m`).
    - For every row with :math:`N_{\\text{act}} \\ge 1` active pairs,
      ``gci1_bot_structural(c, d, margin, agg="sum")`` equals
      :math:`N_{\\text{act}}` times ``gci1_bot_structural(c, d, margin, agg="mean")``.
    - Raises :class:`ValueError` if ``agg`` is not ``"mean"`` or ``"sum"``.

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
    agg : {"mean", "sum"}, default "mean"
        Aggregation over active component pairs: ``"mean"`` averages over
        active pairs; ``"sum"`` sums over active pairs.

    Returns
    -------
    torch.Tensor
        Per-row squared GCI1-bot structural loss of shape ``(B,)``.
    """
    if not isinstance(agg, str) or agg not in ("mean", "sum"):
        raise ValueError(f"agg must be 'mean' or 'sum', got {agg!r}")

    bsz = c.cen.shape[0]
    sov = signed_overlap(c.cen, c.w, d.cen, d.w)
    ne_ij = th.amin(sov, dim=-1).reshape(bsz, -1)
    pair_active = (c.active.unsqueeze(2) & d.active.unsqueeze(1)).reshape(bsz, -1)

    term = th.relu(ne_ij + margin).square()
    num = th.where(pair_active, term, th.zeros_like(term)).sum(dim=1)
    if agg == "sum":
        return num
    mask = pair_active.to(dtype=c.cen.dtype)
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


def witness_loss(
    inside_pos: th.Tensor,
    inside_neg: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Compute the structural witness loss over positive and negative class memberships.

    Evaluates

    .. math::

        \\text{mean}_{(i, D) \\in \\text{pos}} \\text{relu}(m - \\text{inside}(p_i, D))^2
        + \\text{mean}_{(i, D) \\in \\text{neg}} \\text{relu}(\\text{inside}(p_i, D) + m)^2,

    where an empty ``inside_pos`` or ``inside_neg`` tensor contributes ``0.0``.

    Guarantees:

    - Returns a scalar ``torch.Tensor`` of shape ``()``.
    - Equals ``0`` if and only if every entry in ``inside_pos`` is at least
      ``margin`` (:math:`\\text{inside}(p_i, D) \\ge m`, i.e. witness :math:`p_i`
      lies at least :math:`m` inside some active slot of :math:`D`) and every
      entry in ``inside_neg`` is at most ``-margin``
      (:math:`\\text{inside}(p_i, D) \\le -m`, i.e. witness :math:`p_i` lies at
      least :math:`m` outside every active slot of :math:`D`).

    Non-guarantees:

    - Does not deduplicate target rows; callers should pass deduplicated
      targets from :class:`WitnessTargets`.

    Parameters
    ----------
    inside_pos : torch.Tensor
        1-D tensor of shape ``(P,)`` holding :math:`\\text{inside}(p_i, D)` for positive targets.
    inside_neg : torch.Tensor
        1-D tensor of shape ``(N,)`` holding :math:`\\text{inside}(p_i, D)` for negative targets.
    margin : float
        Interior / exterior margin :math:`m`.

    Returns
    -------
    torch.Tensor
        Scalar witness loss.
    """
    dtype = inside_pos.dtype if inside_pos.numel() > 0 else inside_neg.dtype
    device = inside_pos.device if inside_pos.numel() > 0 else inside_neg.device

    if inside_pos.numel() > 0:
        pos_term = th.relu(margin - inside_pos).square().mean()
    else:
        pos_term = th.zeros((), dtype=dtype, device=device)

    if inside_neg.numel() > 0:
        neg_term = th.relu(inside_neg + margin).square().mean()
    else:
        neg_term = th.zeros((), dtype=dtype, device=device)

    return pos_term + neg_term
