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

from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from mowl.evaluation.query_search import (MaxMinBeam, RegionCompiler, WitnessChain,
                                          as_target_table, read_batch)
from mowl.nn.el.grounded.module import GroundedELModule
from mowl.nn.el.grounded.sampling import is_grounded_geometry
from mowl.nn.el.structural import Regions
from mowl.representation.library.box_union_algebra import BoundedBoxUnions
from mowl.representation.library.element_box_family import ElementBoxFamily

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

#: Temperature of the smooth depth of a region (``depth(..., crisp=False)``).
_LSE_TAU: float = 0.1


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


# -- reading the arguments ----------------------------------------------------------------
def _device(module):
    return module.elem_pos.weight.device


def _check(module, translation=False):
    if not is_grounded_geometry(module):
        raise TypeError(f"module must be a GroundedELModule, got {type(module).__name__}")
    if translation and module.role_mode != "translation":
        raise ValueError("GroundedEL query compilation requires role_mode='translation', got "
                         f"{module.role_mode!r}")


def _count(name, value, low):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an int >= {low}, got {value!r}")
    return value


def _finite(name, value):
    if isinstance(value, th.Tensor) and (value.dtype == th.bool or value.is_complex()
                                         or value.numel() != 1):
        raise ValueError(f"{name} must be a finite scalar, got {value!r}")
    if not isinstance(value, th.Tensor) and (isinstance(value, (bool, np.bool_)) or not
                                             isinstance(value, (int, float, np.number))):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    number = float(np.asarray(value.detach().cpu() if isinstance(value, th.Tensor) else value))
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _vector(ids, device, name):
    """``ids`` as a 1-D index tensor (no range check)."""
    if isinstance(ids, th.Tensor):
        if ids.dtype == th.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError(f"{name} must contain integer indices, got dtype={ids.dtype}")
        out = ids.to(device=device, dtype=th.long)
    elif isinstance(ids, (int, np.integer)) and not isinstance(ids, (bool, np.bool_)):
        out = th.tensor([int(ids)], dtype=th.long, device=device)
    elif isinstance(ids, (list, tuple, np.ndarray)):
        arr = np.asarray(ids)
        if arr.size and (arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer)):
            raise ValueError(f"{name} must contain integer indices, got dtype={arr.dtype}")
        out = th.as_tensor(arr.astype(np.int64), device=device)
    else:
        raise TypeError(f"{name} must be an int, Sequence[int], or Tensor, got "
                        f"{type(ids).__name__}")
    if out.ndim > 1:
        raise ValueError(f"{name} must be 1-D (or scalar), got shape {tuple(out.shape)}")
    return out.reshape(-1)


def _boxes_of(region, name, dim=None):
    """``region`` (a :class:`MultiBox` or :class:`Regions`) checked, as a :class:`MultiBox`."""
    if not isinstance(region, (MultiBox, Regions)):
        raise TypeError(f"{name} must be a MultiBox, got {type(region).__name__}")
    parts = (region.cen, region.w, region.active)
    if not all(isinstance(p, th.Tensor) for p in parts):
        raise TypeError(f"{name} fields (cen, w, active) must be torch.Tensors")
    if (region.cen.ndim, region.w.ndim, region.active.ndim) != (3, 3, 2) or \
            region.cen.shape != region.w.shape or region.cen.shape[:2] != region.active.shape:
        raise ValueError(f"Invalid {name} shapes: cen={tuple(region.cen.shape)}, "
                         f"w={tuple(region.w.shape)}, active={tuple(region.active.shape)}")
    if region.active.dtype != th.bool:
        raise ValueError(f"{name}.active must have dtype torch.bool, got {region.active.dtype}")
    if dim is not None and region.cen.shape[2] != dim:
        raise ValueError(f"{name} embed_dim ({region.cen.shape[2]}) does not match module "
                         f"embed_dim ({dim})")
    return region if isinstance(region, MultiBox) else MultiBox(*parts)


def _rows_of(rel_ids, region):
    """One relation per row of ``region`` (a single one is used for every row)."""
    ids = _vector(rel_ids, region.cen.device, "rel_ids")
    rows = region.cen.shape[0]
    if ids.numel() == 1 and rows > 1:
        return ids.expand(rows)
    if ids.numel() != rows:
        raise ValueError(f"Incompatible batch sizes between R ({rows}) and rel_ids "
                         f"({ids.numel()})")
    return ids


def _grid_points(points, region):
    """``points`` ``(d,)``, ``(N, d)`` or ``(B, N, d)`` shaped for the depth in ``region``."""
    bsz, _, dim = region.cen.shape
    if points.ndim == 1 and points.shape[0] == dim:
        return points.reshape(1, 1, 1, dim)
    if points.ndim == 2 and points.shape[1] == dim:
        return points[None, :, None]
    if points.ndim == 3 and points.shape[0] == bsz and points.shape[2] == dim:
        return points[:, :, None]
    if points.ndim not in (1, 2, 3):
        raise ValueError(f"points must have 1, 2, or 3 dimensions, got shape "
                         f"{tuple(points.shape)}")
    raise ValueError(f"points of shape {tuple(points.shape)} do not fit a region of "
                     f"{bsz} rows in dimension d={dim}")


def _family(module, positions=None):
    return ElementBoxFamily(module, positions=positions, box_type=MultiBox, tau=_LSE_TAU)


def _maps(module, entity_to_element, class_of_entity):
    device = _device(module)
    return tuple(None if m is None else _vector(m, device, name)
                 for m, name in ((entity_to_element, "entity_to_element"),
                                 (class_of_entity, "class_of_entity")))


def _batch(queries):
    if not isinstance(queries, (list, tuple)):
        raise TypeError("queries must be a Sequence (list or tuple) of query instances, got "
                        f"{type(queries).__name__}")
    return queries


def _candidates(module, ids):
    if ids is None:
        return None
    pick = _vector(ids, _device(module), "candidate_element_ids")
    if pick.numel() and bool(((pick < 0) | (pick >= module.nb_elements)).any()):
        raise ValueError(f"candidate_element_ids indices out of range [0, {module.nb_elements})")
    return pick


def _witness(module, chain, structure, queries, targets, one):
    """Witness-chain scores of ``targets`` (soft on the positions, witnesses chosen on
    detached ones)."""
    positions = module.positions()
    if not queries or targets.shape[1] == 0:
        shape = (len(queries),) if one else tuple(targets.shape)
        return th.empty(shape, dtype=positions.dtype, device=positions.device)
    scores = chain.scores(_family(module, positions), _family(module, positions.detach()),
                          read_batch(structure, queries), targets, module.tau)
    return scores.squeeze(1) if one else scores


def _hinge_loss(scores, n_pos, pos_one, neg, neg_one, margin, return_scores, module):
    """:math:`[m - s^+]_+` averaged, plus :math:`[m + s^-]_+` averaged for negatives."""
    if scores.shape[0] == 0:
        loss = module.elem_pos.weight.sum() * 0.0
    else:
        loss = th.relu(margin - scores[:, :n_pos]).mean()
    neg_scores = None
    if neg is not None:
        neg_scores = scores[:, n_pos:]
        if neg.shape[1] > 0 and scores.shape[0] > 0:
            loss = loss + th.relu(margin + neg_scores).mean()
        neg_scores = neg_scores.squeeze(1) if neg_one and neg.shape[1] > 0 else neg_scores
    if not return_scores:
        return loss
    pos_scores = scores[:, :n_pos]
    return loss, pos_scores.squeeze(1) if pos_one else pos_scores, neg_scores


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
    _check(module)
    return _family(module).anchor(_vector(entity_element_ids, _device(module),
                                          "entity_element_ids"))


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
    _check(module, translation=True)
    region = _boxes_of(R, "R", module.embed_dim)
    return _family(module).project(region, _rows_of(rel_ids, region),
                                   _count("max_boxes", max_boxes, 1))


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
    a, b = _boxes_of(R1, "R1"), _boxes_of(R2, "R2")
    return BoundedBoxUnions(_count("max_boxes", max_boxes, 1), MultiBox).intersect(a, b)


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
    a, b = _boxes_of(R1, "R1"), _boxes_of(R2, "R2")
    return BoundedBoxUnions(_count("max_boxes", max_boxes, 1), MultiBox).union(a, b)


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
    _check(module)
    ids = None if class_ids is None else _vector(class_ids, _device(module), "class_ids")
    return _family(module).class_region(ids)


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
    region = _boxes_of(R, "R")
    if not isinstance(points, th.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
    if not isinstance(crisp, bool):
        raise TypeError(f"crisp must be a bool, got {type(crisp).__name__}")
    return BoundedBoxUnions(box_type=MultiBox, tau=_LSE_TAU).depth(
        _grid_points(points, region), region, crisp=crisp)


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
    return BoundedBoxUnions(box_type=MultiBox).overlap_depth(_boxes_of(R, "R"),
                                                             _boxes_of(Q, "Q"))


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
    _check(module)
    region = _boxes_of(R, "R", module.embed_dim)
    ids = None if candidate_element_ids is None else \
        _vector(candidate_element_ids, _device(module), "candidate_element_ids")
    return _family(module).entity_scores(region, ids, chunk=_count("chunk", chunk, 1),
                                         max_cells=_count("max_elems", max_elems, 1))


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
    _check(module)
    region = _boxes_of(R, "R", module.embed_dim)
    ids = None if class_ids is None else _vector(class_ids, _device(module), "class_ids")
    return _family(module).concept_scores(region, ids,
                                          max_cells=_count("max_elems", max_elems, 1))


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
    _check(module, translation=True)
    region = _boxes_of(R, "R", module.embed_dim)
    count_w, count_b = _count("max_witnesses", max_witnesses, 1), _count("max_boxes", max_boxes, 1)
    ids = None if element_ids is None else _vector(element_ids, _device(module), "element_ids")
    return _family(module).project_exact(region, _rows_of(rel_ids, region), count_b, ids,
                                         count_w)


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
    _check(module, translation=True)
    count_b, count_w = _count("max_boxes", max_boxes, 1), _count("max_witnesses", max_witnesses, 1)
    if not isinstance(exact, bool):
        raise TypeError(f"exact must be a bool, got {type(exact).__name__}")
    family, maps = _family(module), _maps(module, entity_to_element, class_of_entity)
    if not _batch(queries):
        return family.algebra().empty(0, module.embed_dim, module.positions(th.arange(1)))
    ids = None if element_ids is None else _vector(element_ids, _device(module), "element_ids")
    compiler = RegionCompiler(count_b, exact, ids, count_w, type_rel, type_inv_rel, *maps)
    return compiler.compile(family, read_batch(structure, queries))


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
    _check(module, translation=True)
    beam = MaxMinBeam(_count("k", k, 1), type_rel, type_inv_rel,
                      *_maps(module, entity_to_element, class_of_entity))
    _count("chunk", chunk, 1)
    pick = _candidates(module, candidate_element_ids)
    family = _family(module, module.positions())
    scores = beam.scores(family, read_batch(structure, queries)) if _batch(queries) else \
        th.empty((0, module.nb_elements), dtype=family.dtype, device=family.device)
    return scores if pick is None else scores[:, pick]


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
    _check(module, translation=True)
    chain = WitnessChain(_count("k", k, 1), type_rel, type_inv_rel,
                         *_maps(module, entity_to_element, class_of_entity))
    _count("chunk", chunk, 1)
    _batch(queries)
    targets, one = as_target_table(target_element_ids, len(queries), module.nb_elements,
                                   _device(module), "target_element_ids", allow_empty=True)
    return _witness(module, chain, structure, queries, targets, one)


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
    _check(module, translation=True)
    chain = WitnessChain(_count("k", k, 1), type_rel, type_inv_rel,
                         *_maps(module, entity_to_element, class_of_entity))
    margin = _finite("margin", margin)
    _count("chunk", chunk, 1)
    if not isinstance(return_scores, bool):
        raise TypeError(f"return_scores must be a bool, got {type(return_scores).__name__}")
    bsz, n, device = len(_batch(queries)), module.nb_elements, _device(module)
    pos, pos_one = as_target_table(answers, bsz, n, device, "answers")
    neg, neg_one = (None, False) if negatives is None else \
        as_target_table(negatives, bsz, n, device, "negatives", allow_empty=True)
    both = pos if neg is None else th.cat([pos, neg], dim=1)
    scores = _witness(module, chain, structure, queries, both, False)
    return _hinge_loss(scores, pos.shape[1], pos_one, neg, neg_one, margin, return_scores,
                       module)


def _beam_project_type_inv_rel(module, S, *, k, chunk):
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
    return MaxMinBeam(k).through_classes(_family(module, module.positions()), S)
