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
- **Zero-loss characterisation at margin 0:**
  - :func:`prism_inclusion_score` and :func:`gci0_loss` equal ``0`` iff
    :math:`h_C \\le h_D` coordinatewise and box :math:`B_C \\subseteq B_D`.
  - :func:`gci1_loss` uses the exact closed-form intersection
    (:func:`prism_intersection`, Proposition 5) without component gating.
  - :func:`gci1_bot_loss` equals ``0`` at ``margin = 0`` iff the triangle
    intersection is empty/degenerate (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) \\le 0`)
    or the box part is separated in at least one coordinate.
- **Degenerate / empty triangle safety:** Support vectors with
  :math:`\\sum_k h_k < 0` produce ``0`` in :func:`triangle_empty_score` at
  ``margin = 0`` and never produce ``NaN`` or ``Inf`` in inclusion or GCI losses.

Non-guarantees:
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
    - Returns a 1-D tensor of shape ``(B,)``.
    - At ``margin = 0``, equals ``0`` iff :math:`h_C \\le h_D` in all 3 support
      coordinates and :math:`|b_C - b_D| + |o_C| \\le |o_D|` in all :math:`d-2`
      box coordinates.
    - Finite and differentiable even when :math:`\\sum_k h_{C,k} < 0` or
      :math:`\\sum_k h_{D,k} < 0`.

    Non-guarantees:
    - Does not gate out empty sub-prisms when :math:`\\sum_k h_{C,k} < 0` if
      some individual coordinate still satisfies :math:`h_{C,k} > h_{D,k} - \\text{margin}`.
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
    - At ``margin = 0``, equals ``0`` whenever :math:`\\sum_{k=1}^3 h_k \\le 0`
      (empty or single-point triangle) and equals :math:`\\sum_{k=1}^3 h_k > 0`
      when the triangle has positive inradius.

    Non-guarantees:
    - Does not inspect the box part of a prism.
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
    """GCI0 loss for :math:`C \\sqsubseteq D`."""
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
    """GCI0-bot loss :math:`\\text{relu}(\\sum_k h_{C,k} + \\text{margin})` for :math:`C \\sqsubseteq \\bot`."""
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
    """GCI1 loss for :math:`C \\sqcap D \\sqsubseteq E` via closed-form prism intersection."""
    hc, bc, oc = _unpack_prism(data[:, 0], tri_center, tri_inradius, box_center, box_offset)
    hd, bd, od = _unpack_prism(data[:, 1], tri_center, tri_inradius, box_center, box_offset)
    he, be, oe = _unpack_prism(data[:, 2], tri_center, tri_inradius, box_center, box_offset)

    h_inter, lower, upper = prism_intersection(hc, (bc, oc), hd, (bd, od))
    inter_bc = (lower + upper) / 2.0
    inter_oc = th.abs(upper - lower) / 2.0

    incl = prism_inclusion_score(h_inter, inter_bc, inter_oc, he, be, oe, margin)
    box_disjoint_penalty = th.linalg.norm(th.relu(lower - upper), dim=-1)
    return incl + box_disjoint_penalty


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
    """Positive GCI2 score: inclusion of :math:`C + r` inside :math:`D`."""
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
    """Negative GCI2 score mirroring ELBE's ``gci2_score_neg`` across triangle supports and box coords."""
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
    """GCI2 loss for :math:`C \\sqsubseteq \\exists r.D` (positive and negative)."""
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
    """GCI3 loss for :math:`\\exists r.C \\sqsubseteq D` mirroring ELBE's overlap formula."""
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
    """GCI3-bot loss :math:`\\text{relu}(\\sum_k h_{C,k} + \\text{margin})` for :math:`\\exists r.C \\sqsubseteq \\bot`."""
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
    """Class assertion loss :math:`C(a)` as inclusion of the degenerate point prism :math:`a` in :math:`C`."""
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
    """Object property assertion loss :math:`r(a, b)` for point individuals :math:`a, b`."""
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
