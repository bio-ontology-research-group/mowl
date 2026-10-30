"""Support-coordinate loss functions for Prism embeddings (triangle x box).

Every concept name :math:`C` is represented as a prism
:math:`P_C = T_C \\times B_C \\subset \\mathbb{R}^2 \\times \\mathbb{R}^{d-2}`
following Federico, Leemhuis & Kutz (DL 2026), §4.1 Definition 4 and
Proposition 5:

- :math:`T_C = \\{p \\in \\mathbb{R}^2 : n_k \\cdot p \\le h_{C,k},\\; k=1,2,3\\}` is an
  axis-aligned, apex-up equilateral triangle with fixed unit outward normals
  :math:`n_1 = (0, -1)`, :math:`n_2 = (\\sqrt{3}/2, 1/2)`,
  :math:`n_3 = (-\\sqrt{3}/2, 1/2)` and support values
  :math:`h_{C,k} = n_k \\cdot c_C + r_C` where :math:`c_C \\in \\mathbb{R}^2` is the
  incenter/centroid and :math:`r_C = |\\rho_C| \\ge 0` is the inradius. Since
  :math:`\\sum_{k=1}^3 n_k = 0`, :math:`\\sum_{k=1}^3 h_{C,k} = 3 r_C`, and a
  general support triple defines a non-empty triangle iff
  :math:`\\sum_{k=1}^3 h_k \\ge 0`.
- :math:`B_C = (b_C, |o_C|) \\subset \\mathbb{R}^{d-2}` is an axis-parallel box.
- By Proposition 5 of Federico et al. (DL 2026), prisms are closed under
  intersection: :math:`(T_1 \\times B_1) \\cap (T_2 \\times B_2) = (T_1 \\cap T_2) \\times (B_1 \\cap B_2)`
  where :math:`T_1 \\cap T_2` has support vector :math:`\\min(h_1, h_2)` coordinatewise.

Guarantees:
- **Support-sum identity:** :math:`\\sum_{k=1}^3 \\text{supports}(c, r)_k = 3 r`.
- **Translation equivariance:** :math:`\\text{supports}(c + t, r) = \\text{supports}(c, r) + t \\, \\text{NORMALS}^\\top`.
- **Return shape:** Every per-GCI loss returns a 1-D tensor of shape ``(B,)``
  like MultiBoxEL (whereas ELBE returns ``(B, 1)``).
- **Zero-loss characterisation at margin 0:**
  - :func:`prism_inclusion_score` and :func:`gci0_loss` equal ``0`` iff
    :math:`h_C \\le h_D` coordinatewise and box :math:`B_C \\subseteq B_D`.
  - :func:`gci1_loss` uses the exact closed-form intersection
    (:func:`prism_intersection`, Proposition 5) and adds symmetric non-emptiness
    penalties :math:`\\text{relu}(-\\sum_k \\min(h_{C,k}, h_{D,k}))` on the triangle
    part and :math:`\\|\\text{relu}(\\text{lower} - \\text{upper})\\|_2` on the box part.
  - :func:`gci1_bot_loss` equals ``0`` at ``margin = 0`` iff the triangle
    intersection is empty or degenerate (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) \\le 0`,
    so touching triangles at a single point count as disjoint at ``margin = 0``)
    *or* the box part separates in at least one coordinate via :func:`torch.amin`
    over box coordinates (semantically disjointness-if-any-separates, whereas
    ELBE takes the Euclidean norm over box coordinates and requires separation
    in every coordinate).
- **Degenerate / empty triangle safety:** Support vectors with
  :math:`\\sum_k h_k < 0` produce ``0`` in :func:`triangle_empty_score` at
  ``margin = 0`` and never produce ``NaN`` or ``Inf`` in inclusion or GCI losses.

Non-guarantees:
- **Positive margin floor on single-class bottom losses:** Because
  :math:`\\sum_{k=1}^3 h_{C,k} = 3 |\\rho_C| \\ge 0`, :func:`gci0_bot_loss` and
  :func:`gci3_bot_loss` (:math:`\\text{relu}(3 |\\rho_C| + \\text{margin})`) have a
  lower bound of ``margin`` whenever ``margin > 0`` and cannot reach ``0``.
- **Touching sets at margin 0:** At ``margin = 0``, triangles touching at a
  single point (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) = 0`) and boxes touching at
  a boundary count as disjoint in :func:`gci1_bot_loss` and non-overlapping in
  :func:`gci3_loss`.
- **Trivial GCI1 satisfaction by empty intersection:** :func:`gci1_loss`
  penalises empty triangle or box intersections via
  :math:`\\text{relu}(-\\sum_k \\min(h_{C,k}, h_{D,k})) + \\|\\text{relu}(\\text{lower} - \\text{upper})\\|_2`
  even though :math:`C \\sqcap D = \\emptyset \\subseteq E` holds vacuously.
- Positive ``margin`` tightens constraints following ELBE's sign convention, so
  simultaneous mutual inclusions :math:`A \\sqsubseteq B` and :math:`B \\sqsubseteq A`
  cannot both achieve zero loss when ``margin > 0``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch as th

NORMALS: th.Tensor = th.tensor(
    [
        [0.0, -1.0],
        [math.sqrt(3.0) / 2.0, 0.5],
        [-math.sqrt(3.0) / 2.0, 0.5],
    ],
    dtype=th.float32,
)


def supports(center: th.Tensor, inradius: th.Tensor) -> th.Tensor:
    """Compute triangle support values :math:`h_k = n_k \\cdot c + r` for :math:`k=1,2,3`.

    See Federico, Leemhuis & Kutz (DL 2026), §4.1 Definition 4.

    Guarantees:
    - Returns a tensor of shape ``(..., 3)`` matching the batch prefix of ``center``.
    - ``supports(center, inradius).sum(dim=-1)`` equals ``3 * inradius.squeeze(-1)``.

    Non-guarantees:
    - Does not clamp or take the absolute value of ``inradius``; callers reading
      from an unconstrained embedding table should pass ``torch.abs(raw_inradius)``.

    Parameters
    ----------
    center : torch.Tensor
        Triangle centres of shape ``(..., 2)``.
    inradius : torch.Tensor
        Triangle inradii of shape ``(..., 1)`` or ``(...,)``.

    Returns
    -------
    torch.Tensor
        Support values of shape ``(..., 3)``.
    """
    normals = NORMALS.to(device=center.device, dtype=center.dtype)
    if isinstance(inradius, th.Tensor) and inradius.ndim == center.ndim - 1:
        inradius = inradius.unsqueeze(-1)
    return center @ normals.T + inradius


def _translate_supports(h: th.Tensor, t_tri: th.Tensor) -> th.Tensor:
    """Translate triangle support values ``h`` by 2-D vector(s) ``t_tri`` via ``NORMALS``."""
    normals = NORMALS.to(device=h.device, dtype=h.dtype)
    return h + t_tri @ normals.T


def prism_inclusion_score(
    hc: th.Tensor,
    boxc: th.Tensor,
    boxo: th.Tensor,
    hd: th.Tensor,
    boxd: th.Tensor,
    boxod: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Compute the prism inclusion score for :math:`(T_C \\times B_C) \\subseteq (T_D \\times B_D)`.

    Stacks the 3-component triangle support violation
    :math:`\\text{relu}(h_C - h_D + \\text{margin})` with the :math:`(d-2)`-component
    ELBE box inclusion violation
    :math:`\\text{relu}(|b_C - b_D| + |o_C| - |o_D| + \\text{margin})` and returns the
    Euclidean norm along the last axis.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - At ``margin = 0``, equals ``0`` iff :math:`h_C \\le h_D` in all 3 support
      coordinates and :math:`|b_C - b_D| + |o_C| \\le |o_D|` in all :math:`d-2`
      box coordinates.
    - Finite and differentiable even when :math:`\\sum_k h_{C,k} < 0` or
      :math:`\\sum_k h_{D,k} < 0`.

    Non-guarantees:
    - Does not gate out empty sub-prisms when :math:`\\sum_k h_{C,k} < 0` if
      some individual coordinate still satisfies :math:`h_{C,k} > h_{D,k} - \\text{margin}`.

    Parameters
    ----------
    hc : torch.Tensor
        Support values of :math:`C`, shape ``(B, 3)``.
    boxc : torch.Tensor
        Box centres of :math:`C`, shape ``(B, d - 2)``.
    boxo : torch.Tensor
        Box half-widths of :math:`C`, shape ``(B, d - 2)``.
    hd : torch.Tensor
        Support values of :math:`D`, shape ``(B, 3)``.
    boxd : torch.Tensor
        Box centres of :math:`D`, shape ``(B, d - 2)``.
    boxod : torch.Tensor
        Box half-widths of :math:`D`, shape ``(B, d - 2)``.
    margin : float
        Inclusion margin.

    Returns
    -------
    torch.Tensor
        Per-row score tensor of shape ``(B,)``.
    """
    tri_vec = th.relu(hc - hd + margin)
    box_vec = th.relu(th.abs(boxc - boxd) + th.abs(boxo) - th.abs(boxod) + margin)
    stacked = th.cat([tri_vec, box_vec], dim=-1)
    return th.linalg.norm(stacked, dim=-1)


def prism_intersection(
    h1: th.Tensor,
    b1: Tuple[th.Tensor, th.Tensor],
    h2: th.Tensor,
    b2: Tuple[th.Tensor, th.Tensor],
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Compute the closed-form intersection of two prisms (Proposition 5, Federico et al., DL 2026).

    Guarantees:
    - Returns ``(h, lower, upper)`` with ``h = torch.minimum(h1, h2)`` of shape
      ``(B, 3)`` and box corners ``lower, upper`` of shape ``(B, d - 2)``.
    - The intersection is non-empty iff ``h.sum(dim=-1) >= 0`` and
      ``(lower <= upper).all(dim=-1)``.

    Non-guarantees:
    - When the intersection is empty in the triangle or box part, ``h.sum(dim=-1) < 0``
      or ``lower > upper`` in some coordinate; callers must check non-emptiness
      if required.

    Parameters
    ----------
    h1 : torch.Tensor
        First triangle support tensor of shape ``(B, 3)``.
    b1 : Tuple[torch.Tensor, torch.Tensor]
        First box ``(center, offset)`` of shapes ``(B, d - 2)``.
    h2 : torch.Tensor
        Second triangle support tensor of shape ``(B, 3)``.
    b2 : Tuple[torch.Tensor, torch.Tensor]
        Second box ``(center, offset)`` of shapes ``(B, d - 2)``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple ``(h, lower, upper)`` of intersection support values and box bounds.
    """
    c1, o1 = b1
    c2, o2 = b2
    o1 = th.abs(o1)
    o2 = th.abs(o2)
    h = th.minimum(h1, h2)
    lower = th.maximum(c1 - o1, c2 - o2)
    upper = th.minimum(c1 + o1, c2 + o2)
    return h, lower, upper


def triangle_empty_score(h: th.Tensor, margin: float = 0.0) -> th.Tensor:
    """Penalty :math:`\\text{relu}(\\sum_{k=1}^3 h_k + \\text{margin})` for a triangle to be empty.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` when ``h`` has shape ``(B, 3)``.
    - At ``margin = 0``, equals ``0`` whenever :math:`\\sum_{k=1}^3 h_k \\le 0`
      (empty or single-point touching triangle) and equals :math:`\\sum_{k=1}^3 h_k > 0`
      when the triangle has positive inradius.

    Non-guarantees:
    - At ``margin > 0``, if :math:`\\sum_{k=1}^3 h_k = 3 |\\rho| \\ge 0`, the score
      has a floor of ``margin`` and cannot reach ``0``.
    - Does not inspect the box part of a prism.

    Parameters
    ----------
    h : torch.Tensor
        Triangle support tensor of shape ``(..., 3)``.
    margin : float, default 0.0
        Margin added inside ``relu``.

    Returns
    -------
    torch.Tensor
        Emptiness violation score of shape ``(...,)``.
    """
    return th.relu(h.sum(dim=-1) + margin)


def _unpack_prism(
    idx: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    box_center: th.nn.Embedding,
    box_offset: th.nn.Embedding,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Return ``(h, box_c, box_o)`` for class indices ``idx``."""
    tc = tri_center(idx)
    tr = th.abs(tri_inradius(idx))
    h = supports(tc, tr)
    bc = box_center(idx)
    bo = th.abs(box_offset(idx))
    return h, bc, bo


def gci0_loss(
    data: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    box_center: th.nn.Embedding,
    box_offset: th.nn.Embedding,
    margin: float,
    neg: bool = False,
) -> th.Tensor:
    """GCI0 loss for :math:`C \\sqsubseteq D`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - At ``margin = 0``, equals ``0`` iff :math:`h_C \\le h_D` coordinatewise
      and box :math:`B_C \\subseteq B_D`.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False`` (GCI0 is not in ``neg_capable_gcis``).
    - When ``margin > 0``, simultaneous :math:`C \\sqsubseteq D` and :math:`D \\sqsubseteq C`
      cannot both achieve ``0``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 2)`` with columns ``(C, D)``.
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    box_center : torch.nn.Embedding
        Box centre embedding table of shape ``(N, d - 2)``.
    box_offset : torch.nn.Embedding
        Box half-width embedding table of shape ``(N, d - 2)``.
    margin : float
        Inclusion margin (positive tightens inclusion).
    neg : bool, default False
        Unused flag kept for ``ELModule`` signature compatibility.

    Returns
    -------
    torch.Tensor
        Per-axiom loss tensor of shape ``(B,)``.
    """
    hc, bc, oc = _unpack_prism(data[:, 0], tri_center, tri_inradius, box_center, box_offset)
    hd, bd, od = _unpack_prism(data[:, 1], tri_center, tri_inradius, box_center, box_offset)
    return prism_inclusion_score(hc, bc, oc, hd, bd, od, margin)


def gci0_bot_loss(
    data: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    margin: float = 0.0,
    neg: bool = False,
) -> th.Tensor:
    """GCI0-bot loss :math:`\\text{relu}(\\sum_k h_{C,k} + \\text{margin})` for :math:`C \\sqsubseteq \\bot`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - Since :math:`\\sum_{k=1}^3 h_{C,k} = 3 |\\rho_C| \\ge 0`, equals ``0`` at
      ``margin = 0`` iff :math:`|\\rho_C| = 0` (degenerate single-point triangle).

    Non-guarantees:
    - At ``margin > 0``, because :math:`\\sum_{k=1}^3 h_{C,k} = 3 |\\rho_C| \\ge 0`,
      this loss has a lower bound (floor) of ``margin`` and can never reach ``0``.
    - Does not constrain the box part :math:`B_C`.
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 1)`` or ``(B, 2)`` whose first column is ``C``.
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    margin : float, default 0.0
        Margin added inside ``relu`` before clamping.
    neg : bool, default False
        Unused flag kept for ``ELModule`` signature compatibility.

    Returns
    -------
    torch.Tensor
        Per-axiom loss tensor of shape ``(B,)``.
    """
    tc = tri_center(data[:, 0])
    tr = th.abs(tri_inradius(data[:, 0]))
    hc = supports(tc, tr)
    return triangle_empty_score(hc, margin)


def gci1_loss(
    data: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    box_center: th.nn.Embedding,
    box_offset: th.nn.Embedding,
    margin: float,
    neg: bool = False,
) -> th.Tensor:
    """GCI1 loss for :math:`C \\sqcap D \\sqsubseteq E` via closed-form prism intersection.

    Adds the symmetric empty-intersection penalties
    :math:`\\text{relu}(-\\sum_{k=1}^3 \\min(h_{C,k}, h_{D,k}))` on the triangle part
    and :math:`\\|\\text{relu}(\\text{lower} - \\text{upper})\\|_2` on the box part to
    the prism inclusion score :func:`prism_inclusion_score`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - At ``margin = 0``, equals ``0`` iff :math:`T_C \\cap T_D` is non-empty
      (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) \\ge 0`), :math:`B_C \\cap B_D` is
      non-empty (:math:`\\text{lower} \\le \\text{upper}` coordinatewise), and
      :math:`(T_C \\cap T_D) \\times (B_C \\cap B_D) \\subseteq T_E \\times B_E`.
    - Strictly positive whenever :math:`T_C \\cap T_D` is empty
      (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) < 0`) or :math:`B_C \\cap B_D` is
      empty, even if the remaining coordinates are nested inside :math:`E`.

    Non-guarantees:
    - Penalises empty intersections (:math:`T_C \\cap T_D = \\emptyset` or
      :math:`B_C \\cap B_D = \\emptyset`) even though :math:`C \\sqcap D \\sqsubseteq E`
      holds vacuously when :math:`C \\sqcap D = \\emptyset`.
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(C, D, E)``.
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    box_center : torch.nn.Embedding
        Box centre embedding table of shape ``(N, d - 2)``.
    box_offset : torch.nn.Embedding
        Box half-width embedding table of shape ``(N, d - 2)``.
    margin : float
        Inclusion margin.
    neg : bool, default False
        Unused flag kept for ``ELModule`` signature compatibility.

    Returns
    -------
    torch.Tensor
        Per-axiom loss tensor of shape ``(B,)``.
    """
    hc, bc, oc = _unpack_prism(data[:, 0], tri_center, tri_inradius, box_center, box_offset)
    hd, bd, od = _unpack_prism(data[:, 1], tri_center, tri_inradius, box_center, box_offset)
    he, be, oe = _unpack_prism(data[:, 2], tri_center, tri_inradius, box_center, box_offset)

    h_inter, lower, upper = prism_intersection(hc, (bc, oc), hd, (bd, od))
    inter_bc = (lower + upper) / 2.0
    inter_oc = th.abs(upper - lower) / 2.0

    incl = prism_inclusion_score(h_inter, inter_bc, inter_oc, he, be, oe, margin)
    tri_disjoint_penalty = th.relu(-h_inter.sum(dim=-1))
    box_disjoint_penalty = th.linalg.norm(th.relu(lower - upper), dim=-1)
    return incl + tri_disjoint_penalty + box_disjoint_penalty


def gci1_bot_loss(
    data: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    box_center: th.nn.Embedding,
    box_offset: th.nn.Embedding,
    margin: float,
    neg: bool = False,
) -> th.Tensor:
    """GCI1-bot loss for :math:`C \\sqcap D \\sqsubseteq \\bot`.

    Disjointness holds if the triangle intersection is empty *or* any box
    coordinate separates:
    :math:`\\min(\\text{relu}(\\sum_k \\min(h_{C,k}, h_{D,k}) + \\text{margin}),\\; \\min_m \\text{relu}(-|b_{C,m} - b_{D,m}| + o_{C,m} + o_{D,m} + \\text{margin}))`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - Uses :func:`torch.amin` over box coordinates (semantically
      disjointness-if-any-separates), unlike ELBE which takes the Euclidean norm
      over box coordinates and requires separation in every coordinate.
    - At ``margin = 0``, equals ``0`` iff :math:`\\sum_k \\min(h_{C,k}, h_{D,k}) \\le 0`
      or :math:`|b_{C,m} - b_{D,m}| \\ge o_{C,m} + o_{D,m}` for at least one
      box coordinate :math:`m`.

    Non-guarantees:
    - At ``margin = 0``, touching triangles (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) = 0`,
      intersecting at a single point) and boxes touching at a boundary count as
      disjoint.
    - Taking ``min`` across the triangle term and box coordinates routes gradient
      only through the single active minimum branch at each step.
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 2)`` or ``(B, 3)`` whose first two columns are ``(C, D)``.
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    box_center : torch.nn.Embedding
        Box centre embedding table of shape ``(N, d - 2)``.
    box_offset : torch.nn.Embedding
        Box half-width embedding table of shape ``(N, d - 2)``.
    margin : float
        Disjointness margin (positive requires strict separation).
    neg : bool, default False
        Unused flag kept for ``ELModule`` signature compatibility.

    Returns
    -------
    torch.Tensor
        Per-axiom loss tensor of shape ``(B,)``.
    """
    hc, bc, oc = _unpack_prism(data[:, 0], tri_center, tri_inradius, box_center, box_offset)
    hd, bd, od = _unpack_prism(data[:, 1], tri_center, tri_inradius, box_center, box_offset)

    tri_term = triangle_empty_score(th.minimum(hc, hd), margin)
    box_coord_terms = th.relu(-th.abs(bc - bd) + oc + od + margin)
    min_box_term = th.amin(box_coord_terms, dim=-1)
    return th.minimum(tri_term, min_box_term)


def gci2_score(
    hc: th.Tensor,
    bc: th.Tensor,
    oc: th.Tensor,
    hd: th.Tensor,
    bd: th.Tensor,
    od: th.Tensor,
    rel: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Positive GCI2 score: inclusion of :math:`C + r` inside :math:`D`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - At ``margin = 0``, equals ``0`` iff the translated prism :math:`P_C + r`
      is contained in :math:`P_D`.

    Non-guarantees:
    - Role translation is a single vector shift and cannot express role
      fan-out (one-to-many relations) without forcing target prisms to grow.

    Parameters
    ----------
    hc : torch.Tensor
        Support values of :math:`C`, shape ``(B, 3)``.
    bc : torch.Tensor
        Box centres of :math:`C`, shape ``(B, d - 2)``.
    oc : torch.Tensor
        Box half-widths of :math:`C`, shape ``(B, d - 2)``.
    hd : torch.Tensor
        Support values of :math:`D`, shape ``(B, 3)``.
    bd : torch.Tensor
        Box centres of :math:`D`, shape ``(B, d - 2)``.
    od : torch.Tensor
        Box half-widths of :math:`D`, shape ``(B, d - 2)``.
    rel : torch.Tensor
        Role translation vectors of shape ``(B, d)``.
    margin : float
        Inclusion margin.

    Returns
    -------
    torch.Tensor
        Per-row score tensor of shape ``(B,)``.
    """
    r_tri = rel[:, :2]
    r_box = rel[:, 2:]
    hc_trans = _translate_supports(hc, r_tri)
    bc_trans = bc + r_box
    return prism_inclusion_score(hc_trans, bc_trans, oc, hd, bd, od, margin)


def gci2_score_neg(
    hc: th.Tensor,
    rc: th.Tensor,
    bc: th.Tensor,
    oc: th.Tensor,
    hd: th.Tensor,
    bd: th.Tensor,
    od: th.Tensor,
    rel: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Negative GCI2 score mirroring ELBE's ``gci2_score_neg`` across triangle supports and box coords.

    Since :math:`h_{C+r,k} - 2 |r_C| - h_{D,k} = (c_C + r_{\\text{tri}} - c_D) \\cdot n_k - |r_C| - |r_D|`,
    the triangle component is :math:`\\text{relu}(h_{C+r} - 2 |r_C| - h_D - \\text{margin})`
    and the box component is :math:`\\text{relu}(|b_C + r_{\\text{box}} - b_D| - |o_C| - |o_D| - \\text{margin})`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - Finite and differentiable for all :math:`r_C \\ge 0`, including point
      individuals (:math:`r_C = 0`).

    Non-guarantees:
    - Like ELBE's ``gci2_score_neg``, this score measures separation Violation
      (zero when :math:`C + r` is close to :math:`D`, positive when far apart)
      for downstream negative-sampling losses.

    Parameters
    ----------
    hc : torch.Tensor
        Support values of :math:`C`, shape ``(B, 3)``.
    rc : torch.Tensor
        Inradii of :math:`C`, shape ``(B, 1)``.
    bc : torch.Tensor
        Box centres of :math:`C`, shape ``(B, d - 2)``.
    oc : torch.Tensor
        Box half-widths of :math:`C`, shape ``(B, d - 2)``.
    hd : torch.Tensor
        Support values of :math:`D`, shape ``(B, 3)``.
    bd : torch.Tensor
        Box centres of :math:`D`, shape ``(B, d - 2)``.
    od : torch.Tensor
        Box half-widths of :math:`D`, shape ``(B, d - 2)``.
    rel : torch.Tensor
        Role translation vectors of shape ``(B, d)``.
    margin : float
        Margin subtracted inside ``relu``.

    Returns
    -------
    torch.Tensor
        Per-row negative score tensor of shape ``(B,)``.
    """
    r_tri = rel[:, :2]
    r_box = rel[:, 2:]
    hc_trans = _translate_supports(hc, r_tri)
    tri_vec = th.relu(hc_trans - 2.0 * th.abs(rc) - hd - margin)
    box_vec = th.relu(th.abs(bc + r_box - bd) - th.abs(oc) - th.abs(od) - margin)
    return th.linalg.norm(th.cat([tri_vec, box_vec], dim=-1), dim=-1)


def gci2_loss(
    data: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    box_center: th.nn.Embedding,
    box_offset: th.nn.Embedding,
    rel_embed: th.nn.Embedding,
    margin: float,
    neg: bool = False,
) -> th.Tensor:
    """GCI2 loss for :math:`C \\sqsubseteq \\exists r.D` (positive and negative).

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - When ``neg=False`` and ``margin = 0``, equals ``0`` iff :math:`P_C + r \\subseteq P_D`.
    - When ``neg=True``, dispatches to :func:`gci2_score_neg`.

    Non-guarantees:
    - When ``margin > 0``, positive loss requires :math:`P_C + r` to be strictly
      inside :math:`P_D` by ``margin``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(C, r, D)``.
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    box_center : torch.nn.Embedding
        Box centre embedding table of shape ``(N, d - 2)``.
    box_offset : torch.nn.Embedding
        Box half-width embedding table of shape ``(N, d - 2)``.
    rel_embed : torch.nn.Embedding
        Relation translation embedding table of shape ``(R, d)``.
    margin : float
        Inclusion / separation margin.
    neg : bool, default False
        If ``True``, compute :func:`gci2_score_neg`.

    Returns
    -------
    torch.Tensor
        Per-axiom loss tensor of shape ``(B,)``.
    """
    hc, bc, oc = _unpack_prism(data[:, 0], tri_center, tri_inradius, box_center, box_offset)
    hd, bd, od = _unpack_prism(data[:, 2], tri_center, tri_inradius, box_center, box_offset)
    rel = rel_embed(data[:, 1])
    if neg:
        rc = th.abs(tri_inradius(data[:, 0]))
        return gci2_score_neg(hc, rc, bc, oc, hd, bd, od, rel, margin)
    return gci2_score(hc, bc, oc, hd, bd, od, rel, margin)


def gci3_loss(
    data: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    box_center: th.nn.Embedding,
    box_offset: th.nn.Embedding,
    rel_embed: th.nn.Embedding,
    margin: float,
    neg: bool = False,
) -> th.Tensor:
    """GCI3 loss for :math:`\\exists r.C \\sqsubseteq D` mirroring ELBE's overlap formula.

    Combines the triangle overlap violation
    :math:`\\text{relu}(-\\sum_{k=1}^3 \\min(h_{C,k} - n_k \\cdot r_{\\text{tri}}, h_{D,k}) + \\text{margin})`
    with ELBE's box overlap violation
    :math:`\\text{relu}(|b_C - r_{\\text{box}} - b_D| - o_C - o_D + \\text{margin})`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - At ``margin = 0``, equals ``0`` iff the shifted prism :math:`P_C - r` has a
      non-empty intersection with :math:`P_D` (i.e. :math:`\\sum_k \\min(h_{C-r,k}, h_{D,k}) \\ge 0`
      and :math:`|b_C - r_{\\text{box}} - b_D| \\le o_C + o_D` coordinatewise).

    Non-guarantees:
    - Following ELBE's ``gci3_loss``, this formula only requires :math:`P_C - r`
      to overlap :math:`P_D`, which is a relaxation of full existential inclusion
      when roles are not functional.
    - At ``margin = 0``, touching triangles (:math:`\\sum_k \\min(h_{C-r,k}, h_{D,k}) = 0`)
      achieve ``0`` loss; at ``margin > 0`` they incur positive loss.
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(r, C, D)``.
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    box_center : torch.nn.Embedding
        Box centre embedding table of shape ``(N, d - 2)``.
    box_offset : torch.nn.Embedding
        Box half-width embedding table of shape ``(N, d - 2)``.
    rel_embed : torch.nn.Embedding
        Relation translation embedding table of shape ``(R, d)``.
    margin : float
        Overlap margin.
    neg : bool, default False
        Unused flag kept for ``ELModule`` signature compatibility.

    Returns
    -------
    torch.Tensor
        Per-axiom loss tensor of shape ``(B,)``.
    """
    rel = rel_embed(data[:, 0])
    hc, bc, oc = _unpack_prism(data[:, 1], tri_center, tri_inradius, box_center, box_offset)
    hd, bd, od = _unpack_prism(data[:, 2], tri_center, tri_inradius, box_center, box_offset)

    r_tri = rel[:, :2]
    r_box = rel[:, 2:]
    hc_shifted = _translate_supports(hc, -r_tri)
    tri_term = th.relu(-th.minimum(hc_shifted, hd).sum(dim=-1) + margin).unsqueeze(-1)
    box_term = th.relu(th.abs(bc - r_box - bd) - oc - od + margin)
    return th.linalg.norm(th.cat([tri_term, box_term], dim=-1), dim=-1)


def gci3_bot_loss(
    data: th.Tensor,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    margin: float = 0.0,
    neg: bool = False,
) -> th.Tensor:
    """GCI3-bot loss :math:`\\text{relu}(\\sum_k h_{C,k} + \\text{margin})` for :math:`\\exists r.C \\sqsubseteq \\bot`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - Since :math:`\\sum_{k=1}^3 h_{C,k} = 3 |\\rho_C| \\ge 0`, equals ``0`` at
      ``margin = 0`` iff :math:`|\\rho_C| = 0`.

    Non-guarantees:
    - At ``margin > 0``, because :math:`\\sum_{k=1}^3 h_{C,k} = 3 |\\rho_C| \\ge 0`,
      this loss has a lower bound (floor) of ``margin`` and can never reach ``0``.
    - Mirroring ELBE's ``gci3_bot_loss``, shrinks :math:`C` itself rather than
      modelling role domain restrictions separately.
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 2)`` or ``(B, 3)`` where column 1 is ``C``
        (or shape ``(B, 1)`` where column 0 is ``C``).
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    margin : float, default 0.0
        Margin added inside ``relu`` before clamping.
    neg : bool, default False
        Unused flag kept for ``ELModule`` signature compatibility.

    Returns
    -------
    torch.Tensor
        Per-axiom loss tensor of shape ``(B,)``.
    """
    col = 1 if data.shape[1] > 1 else 0
    tc = tri_center(data[:, col])
    tr = th.abs(tri_inradius(data[:, col]))
    hc = supports(tc, tr)
    return triangle_empty_score(hc, margin)


def class_assertion_loss(
    data: th.Tensor,
    ind_embed: th.nn.Embedding,
    tri_center: th.nn.Embedding,
    tri_inradius: th.nn.Embedding,
    box_center: th.nn.Embedding,
    box_offset: th.nn.Embedding,
    margin: float,
    neg: bool = False,
) -> th.Tensor:
    """Class assertion loss :math:`C(a)` as inclusion of the degenerate point prism :math:`a` in :math:`C`.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - At ``margin = 0``, equals ``0`` iff the point :math:`x_a = (x_{\\text{tri}}, x_{\\text{box}})`
      lies inside triangle :math:`T_C` (:math:`n_k \\cdot x_{\\text{tri}} \\le h_{C,k}`)
      and inside box :math:`B_C` (:math:`|x_{\\text{box}} - b_C| \\le o_C`).

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False`` (``class_assertion`` is not
      in ``neg_capable_gcis``).

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 2)`` with columns ``(a, C)``.
    ind_embed : torch.nn.Embedding
        Individual point embedding table of shape ``(N_ind, d)``.
    tri_center : torch.nn.Embedding
        Triangle centre embedding table of shape ``(N, 2)``.
    tri_inradius : torch.nn.Embedding
        Triangle inradius embedding table of shape ``(N, 1)``.
    box_center : torch.nn.Embedding
        Box centre embedding table of shape ``(N, d - 2)``.
    box_offset : torch.nn.Embedding
        Box half-width embedding table of shape ``(N, d - 2)``.
    margin : float
        Inclusion margin.
    neg : bool, default False
        Unused flag kept for ``ELModule`` signature compatibility.

    Returns
    -------
    torch.Tensor
        Per-assertion loss tensor of shape ``(B,)``.
    """
    x = ind_embed(data[:, 0])
    x_tri = x[:, :2]
    x_box = x[:, 2:]
    zero_r = th.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    hi = supports(x_tri, zero_r)
    oi = th.zeros_like(x_box)

    hc, bc, oc = _unpack_prism(data[:, 1], tri_center, tri_inradius, box_center, box_offset)
    return prism_inclusion_score(hi, x_box, oi, hc, bc, oc, margin)


def object_property_assertion_loss(
    data: th.Tensor,
    ind_embed: th.nn.Embedding,
    rel_embed: th.nn.Embedding,
    margin: float,
    neg: bool = False,
) -> th.Tensor:
    """Object property assertion loss :math:`r(a, b)` for point individuals :math:`a, b`.

    Evaluates :func:`gci2_score` (or :func:`gci2_score_neg` when ``neg=True``)
    on degenerate point prisms with zero inradius and zero box offsets.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)`` like MultiBoxEL (ELBE returns ``(B, 1)``).
    - When ``neg=False`` and ``margin = 0``, equals ``0`` iff :math:`x_a + r = x_b`
      (since :math:`\\sum_k n_k = 0`, :math:`n_k \\cdot (x_{a,\\text{tri}} + r_{\\text{tri}} - x_{b,\\text{tri}}) \\le 0`
      for all :math:`k=1,2,3` iff :math:`x_{a,\\text{tri}} + r_{\\text{tri}} = x_{b,\\text{tri}}`).
    - Supports ``neg=True`` via :func:`gci2_score_neg`.

    Non-guarantees:
    - When ``neg=False`` and ``margin > 0``, because :math:`\\sum_k n_k = 0`, at
      least one support coordinate of :math:`x_a + r - x_b` is non-negative, so
      the triangle term has a positive floor proportional to ``margin``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(a, r, b)``.
    ind_embed : torch.nn.Embedding
        Individual point embedding table of shape ``(N_ind, d)``.
    rel_embed : torch.nn.Embedding
        Relation translation embedding table of shape ``(R, d)``.
    margin : float
        Inclusion / separation margin.
    neg : bool, default False
        If ``True``, compute :func:`gci2_score_neg`.

    Returns
    -------
    torch.Tensor
        Per-assertion loss tensor of shape ``(B,)``.
    """
    x1 = ind_embed(data[:, 0])
    rel = rel_embed(data[:, 1])
    x2 = ind_embed(data[:, 2])

    zero_r = th.zeros((x1.shape[0], 1), device=x1.device, dtype=x1.dtype)
    h1 = supports(x1[:, :2], zero_r)
    b1 = x1[:, 2:]
    o1 = th.zeros_like(b1)

    h2 = supports(x2[:, :2], zero_r)
    b2 = x2[:, 2:]
    o2 = th.zeros_like(b2)

    if neg:
        return gci2_score_neg(h1, zero_r, b1, o1, h2, b2, o2, rel, margin)
    return gci2_score(h1, b1, o1, h2, b2, o2, rel, margin)
