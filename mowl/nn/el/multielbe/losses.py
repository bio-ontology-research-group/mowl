"""Loss functions for MultiELBE.

Every concept name :math:`C` is represented as a union of :math:`k` axis-parallel
boxes :math:`C = \\bigcup_{i=1}^k C_i` with :math:`C_i = (c_{C,i}, |o_{C,i}|) \\subset \\mathbb{R}^d`.
Roles :math:`r` are represented by single translation vectors :math:`r \\in \\mathbb{R}^d`
as in ELBE, and individuals :math:`a` are represented by single boxes as in ELBE.

Guarantees:
- **Exact k = 1 identity with ELBE:** When ``n_components == 1``, every loss
  function in this module (``gci0_loss``, ``gci0_bot_loss``, ``gci1_loss``,
  ``gci1_bot_loss``, ``gci2_loss`` positive and negative, ``gci3_loss``,
  ``gci3_bot_loss``, ``class_assertion_loss``, ``object_property_assertion_loss``)
  reduces algebraically and numerically (to floating-point tolerance ``1e-6``)
  to the corresponding function in :mod:`mowl.nn.el.elbe.losses`, including on
  ``gci1`` batches whose pairwise intersections are empty and preserving ELBE's
  margin sign convention (positive ``margin`` tightens).
- **Zero-loss characterisation at margin 0:** At ``margin = 0``:
  - ``gci0_loss`` is zero if and only if each component :math:`C_i` is contained
    in at least one component :math:`D_j`.
  - ``gci1_loss`` is zero when at least one pair :math:`(C_i, D_j)` has a
    non-empty intersection (:math:`\\text{lo}_{ij} \\le \\text{up}_{ij}`) and every
    pair with non-empty intersection is contained in some component :math:`E_m`;
    disjoint component pairs :math:`(C_i, D_j)` are gated out and impose no
    inclusion penalty.
  - ``gci3_loss`` preserves ELBE's overlap formula
    :math:`\\|\\text{relu}(|c - r - d| - o_c - o_d + \\text{margin})\\|_2` rather
    than full box inclusion.

Non-guarantees:
- **Disjointness vs. coordinatewise separation in ``gci1_bot_loss``:** Two boxes
  that are disjoint in :math:`\\mathbb{R}^d` because they are separated along one
  coordinate still incur a positive ``gci1_bot_loss`` if their projections
  overlap along any other coordinate.
- **Non-convexity across components:** Taking minima over target components and
  gating over pairwise intersections makes the objective piecewise smooth rather
  than convex; gradient descent is not guaranteed to reach a global minimum from
  arbitrary random initialisations.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch as th

import mowl.nn.el.elbe.losses as EL
from mowl.nn.el.multiboxel.losses import (
    _resolve_n_components,
    _unpack_boxes,
    _validate_aggregation,
    multi_intersection,
)
from mowl.nn.el.softagg import soft_weights, softmin


__all__ = [
    "_resolve_n_components",
    "_unpack_boxes",
    "multi_intersection",
    "multi_pair_score",
    "multi_inclusion",
    "gci0_loss",
    "gci0_bot_loss",
    "gci1_loss",
    "gci1_bot_loss",
    "gci2_score",
    "gci2_loss",
    "gci3_pair_score",
    "gci3_loss",
    "gci3_bot_loss",
    "class_assertion_loss",
    "object_property_assertion_loss",
]


def multi_pair_score(
    fn: Callable[..., th.Tensor],
    cen_a: th.Tensor,
    off_a: th.Tensor,
    cen_b: th.Tensor,
    off_b: th.Tensor,
    *args: Any,
    **kwargs: Any,
) -> th.Tensor:
    """Evaluate a pairwise ELBE box score ``fn`` over all :math:`k_a \\times k_b` component pairs.

    Broadcasts ``(cen_a, |off_a|)`` of shape ``(B, ka, d)`` against ``(cen_b, |off_b|)``
    of shape ``(B, kb, d)`` (along with any extra per-sample vector arguments of
    shape ``(B, d)``, such as role translations ``rel``) to ``(B * ka * kb, d)``,
    calls ``fn``, and reshapes the output to ``(B, ka, kb)``.

    Parameters
    ----------
    fn : callable
        Pairwise score function accepting ``(c_a, off_a, c_b, off_b, *args, **kwargs)``
        with 2-D tensors of shape ``(N, d)`` and returning shape ``(N, 1)`` or ``(N,)``.
    cen_a : torch.Tensor
        Component centres of shape ``(B, ka, d)``.
    off_a : torch.Tensor
        Component half-widths of shape ``(B, ka, d)``.
    cen_b : torch.Tensor
        Component centres of shape ``(B, kb, d)``.
    off_b : torch.Tensor
        Component half-widths of shape ``(B, kb, d)``.
    *args, **kwargs
        Additional arguments forwarded to ``fn``. Any ``torch.Tensor`` of shape
        ``(B, d)`` is broadcast across the ``(ka, kb)`` component grid.

    Returns
    -------
    torch.Tensor
        Pairwise scores of shape ``(B, ka, kb)``.
    """
    b_size, ka, d = cen_a.shape
    kb = cen_b.shape[1]
    n_flat = b_size * ka * kb

    ca = cen_a.unsqueeze(2).expand(b_size, ka, kb, d).reshape(n_flat, d)
    oa = th.abs(off_a).unsqueeze(2).expand(b_size, ka, kb, d).reshape(n_flat, d)
    cb = cen_b.unsqueeze(1).expand(b_size, ka, kb, d).reshape(n_flat, d)
    ob = th.abs(off_b).unsqueeze(1).expand(b_size, ka, kb, d).reshape(n_flat, d)

    def _expand_arg(val: Any) -> Any:
        if isinstance(val, th.Tensor) and val.dim() == 2 and val.shape == (b_size, d):
            return val.unsqueeze(1).unsqueeze(2).expand(b_size, ka, kb, d).reshape(n_flat, d)
        return val

    exp_args = [_expand_arg(a) for a in args]
    exp_kwargs = {k: _expand_arg(v) for k, v in kwargs.items()}
    out = fn(ca, oa, cb, ob, *exp_args, **exp_kwargs)
    return out.reshape(b_size, ka, kb)


def multi_inclusion(
    cen_a: th.Tensor,
    off_a: th.Tensor,
    cen_b: th.Tensor,
    off_b: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Compute pairwise ELBE inclusion scores between two component sets.

    Delegates to :func:`mowl.nn.el.elbe.losses.box_inclusion_score` via
    :func:`multi_pair_score`.

    Guarantees:
    - For ``ka = kb = 1``, ``multi_inclusion(cen_a, off_a, cen_b, off_b, margin)[:, 0, 0]``
      equals :func:`mowl.nn.el.elbe.losses.box_inclusion_score` on the squeezed boxes.
    - At ``margin = 0``, entry ``(b, i, j)`` is ``0`` if and only if box ``A_{b,i}``
      is contained in box ``B_{b,j}``.

    Parameters
    ----------
    cen_a : torch.Tensor
        Component centres of shape ``(B, ka, d)``.
    off_a : torch.Tensor
        Component half-widths of shape ``(B, ka, d)``.
    cen_b : torch.Tensor
        Component centres of shape ``(B, kb, d)``.
    off_b : torch.Tensor
        Component half-widths of shape ``(B, kb, d)``.
    margin : float
        ELBE inclusion margin (positive margin tightens).

    Returns
    -------
    torch.Tensor
        Pairwise inclusion scores of shape ``(B, ka, kb)``.
    """
    return multi_pair_score(EL.box_inclusion_score, cen_a, off_a, cen_b, off_b, margin)


def gci0_loss(
    data: th.Tensor,
    class_embed: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    margin: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE GCI0 loss :math:`\\text{mean}_i \\text{agg}_j \\text{incl}(C_i, D_j)` for :math:`C \\sqsubseteq D`.

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci0_loss` at ``k = 1``
      under both ``"hard"`` and ``"soft"``.
    - Converges from ``aggregation="soft"`` to ``aggregation="hard"`` as ``tau -> 0+``.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    """
    _validate_aggregation(aggregation, tau)
    cen_c, off_c = _unpack_boxes(data[:, 0], class_embed, class_offset, n_components)
    cen_d, off_d = _unpack_boxes(data[:, 1], class_embed, class_offset, n_components)
    incl = multi_inclusion(cen_c, off_c, cen_d, off_d, margin)
    if aggregation == "hard":
        return th.amin(incl, dim=2).mean(dim=1, keepdim=True)
    return softmin(incl, tau=tau, dim=2).mean(dim=1, keepdim=True)


def gci0_bot_loss(
    data: th.Tensor,
    class_offset: th.nn.Embedding,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE GCI0-bot loss :math:`\\text{mean}_i \\|o_{C_i}\\|_2` for :math:`C \\sqsubseteq \\bot`.

    Returns shape ``(B,)``, matching :func:`mowl.nn.el.elbe.losses.gci0_bot_loss` at ``k = 1``.
    """
    _validate_aggregation(aggregation, tau)
    b_size = data.shape[0]
    k = _resolve_n_components(class_offset, n_components)
    raw_o = th.abs(class_offset(data[:, 0]))
    d = raw_o.shape[-1] // k
    off_c = raw_o.view(b_size, k, d)
    return th.linalg.norm(off_c, dim=-1).mean(dim=1)


def gci1_loss(
    data: th.Tensor,
    class_embed: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    margin: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Gated ELBE GCI1 loss for :math:`C \\sqcap D \\sqsubseteq E`.

    Uses the same pairwise intersection gating rule as :mod:`mowl.nn.el.multiboxel.losses`:
    - Under ``aggregation="hard"``:
      :math:`(i^*, j^*) = \\arg\\min_{i,j} \\text{empt}(C_i, D_j)` (chosen on detached
      emptiness penalties), gate :math:`g_{ij} = 1` if :math:`\\text{lo}_{ij} \\le \\text{up}_{ij}`
      in every coordinate or :math:`(i, j) = (i^*, j^*)` else :math:`0` (detached), and
      score :math:`= \\frac{\\sum_{i,j} g_{ij} \\min_m \\text{incl}(I(C_i, D_j), E_m)}{\\sum_{i,j} g_{ij}} + \\text{empt}(C_{i^*}, D_{j^*})`.
    - Under ``aggregation="soft"``:
      :math:`w_{ij} = \\text{softmax}(-\\text{empt}_{ij} / \\tau)` (with gradient),
      emptiness term :math:`\\sum_{i,j} w_{ij} \\text{empt}_{ij}`, gate
      :math:`g_{ij} = \\max(\\text{non\\_empty}_{ij}^\\text{det}, w_{ij})`, and inner
      target aggregation :math:`\\text{softmin}_\\tau` over :math:`E_m`.

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci1_loss` at ``k = 1``
      on both empty and non-empty intersections under both ``"hard"`` and ``"soft"``.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    """
    _validate_aggregation(aggregation, tau)
    b_size = data.shape[0]
    cen_c, off_c = _unpack_boxes(data[:, 0], class_embed, class_offset, n_components)
    cen_d, off_d = _unpack_boxes(data[:, 1], class_embed, class_offset, n_components)
    cen_e, off_e = _unpack_boxes(data[:, 2], class_embed, class_offset, n_components)
    k = cen_c.shape[1]

    lower, upper = multi_intersection(cen_c, off_c, cen_d, off_d)
    inter_cen = (lower + upper) / 2
    inter_off = th.abs(lower - upper) / 2

    empt = th.linalg.norm(th.relu(lower - upper), dim=-1)  # (B, k, k)
    empt_flat = empt.view(b_size, k * k)

    ic = inter_cen.unsqueeze(3)  # (B, k, k, 1, d)
    io = inter_off.unsqueeze(3)  # (B, k, k, 1, d)
    ce = cen_e.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k, d)
    oe = off_e.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k, d)
    euc_ie = th.abs(ic - ce)
    incl_ijm = th.linalg.norm(th.relu(euc_ie + io - oe + margin), dim=-1)  # (B, k, k, k)

    if aggregation == "hard":
        designated_idx = th.argmin(empt_flat.detach(), dim=1)  # (B,)
        batch_arange = th.arange(b_size, device=empt.device)
        empt_designated = empt_flat[batch_arange, designated_idx]  # (B,)

        non_empty = (lower.detach() <= upper.detach()).all(dim=-1)  # (B, k, k)
        desig_mask = th.zeros((b_size, k * k), dtype=th.bool, device=empt.device)
        desig_mask[batch_arange, designated_idx] = True
        desig_mask = desig_mask.view(b_size, k, k)
        gate = (non_empty | desig_mask).to(dtype=empt.dtype).detach()  # (B, k, k)

        min_incl_ij = th.amin(incl_ijm, dim=3)  # (B, k, k)
        gated_incl = (gate * min_incl_ij).sum(dim=(1, 2)) / gate.sum(dim=(1, 2))
        return (gated_incl + empt_designated).unsqueeze(1)

    w_flat = soft_weights(empt_flat, tau=tau, dim=1)  # (B, k * k)
    empt_soft = (w_flat * empt_flat).sum(dim=1)  # (B,)
    w = w_flat.view(b_size, k, k)  # (B, k, k)

    non_empty_f = (
        (lower.detach() <= upper.detach())
        .all(dim=-1)
        .to(dtype=empt.dtype)
        .detach()
    )  # (B, k, k)
    gate = th.maximum(non_empty_f, w)  # (B, k, k)

    min_incl_ij = softmin(incl_ijm, tau=tau, dim=3)  # (B, k, k)
    gated_incl = (gate * min_incl_ij).sum(dim=(1, 2)) / gate.sum(dim=(1, 2))
    return (gated_incl + empt_soft).unsqueeze(1)


def _gci1_bot_pair_score(
    c: th.Tensor,
    off_c: th.Tensor,
    d: th.Tensor,
    off_d: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Pairwise ELBE GCI1-bot disjointness score :math:`\\|\\text{relu}(-|c - d| + o_c + o_d + \\text{margin})\\|_2`."""
    euc = th.abs(c - d)
    return th.reshape(
        th.linalg.norm(th.relu(-euc + off_c + off_d + margin), axis=1), [-1, 1]
    )


def gci1_bot_loss(
    data: th.Tensor,
    class_embed: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    margin: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE GCI1-bot loss :math:`\\text{mean}_{i,j} \\|\\text{relu}(-|c_{C,i} - c_{D,j}| + o_{C,i} + o_{D,j} + \\text{margin})\\|_2` for :math:`C \\sqcap D \\sqsubseteq \\bot`.

    Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci1_bot_loss` at ``k = 1``.
    """
    _validate_aggregation(aggregation, tau)
    cen_c, off_c = _unpack_boxes(data[:, 0], class_embed, class_offset, n_components)
    cen_d, off_d = _unpack_boxes(data[:, 1], class_embed, class_offset, n_components)
    pair_scores = multi_pair_score(_gci1_bot_pair_score, cen_c, off_c, cen_d, off_d, margin)
    return pair_scores.mean(dim=(1, 2)).unsqueeze(1)


def gci2_score(
    data: th.Tensor,
    class_embed: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    rel_embed: th.nn.Embedding,
    margin: float,
    n_components: Optional[int] = None,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Positive ELBE GCI2 score for :math:`C \\sqsubseteq \\exists r.D`."""
    return gci2_loss(
        data,
        class_embed,
        class_offset,
        rel_embed,
        margin,
        n_components=n_components,
        neg=False,
        aggregation=aggregation,
        tau=tau,
    )


def gci2_loss(
    data: th.Tensor,
    class_embed: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    rel_embed: th.nn.Embedding,
    margin: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE GCI2 loss :math:`\\text{mean}_i \\text{agg}_j s(C_i, D_j, r)` for :math:`C \\sqsubseteq \\exists r.D` (positive and negative).

    Uses :func:`mowl.nn.el.elbe.losses.gci2_score` when ``neg=False`` and
    :func:`mowl.nn.el.elbe.losses.gci2_score_neg` when ``neg=True``.
    Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci2_loss` at ``k = 1``.
    """
    _validate_aggregation(aggregation, tau)
    d_single = int(rel_embed.embedding_dim)
    cen_c, off_c = _unpack_boxes(
        data[:, 0], class_embed, class_offset, n_components, embed_dim=d_single
    )
    r = rel_embed(data[:, 1])
    cen_d, off_d = _unpack_boxes(
        data[:, 2], class_embed, class_offset, n_components, embed_dim=d_single
    )
    fn = EL.gci2_score_neg if neg else EL.gci2_score
    pair_scores = multi_pair_score(fn, cen_c, off_c, cen_d, off_d, r, margin)
    if aggregation == "hard":
        return th.amin(pair_scores, dim=2).mean(dim=1, keepdim=True)
    return softmin(pair_scores, tau=tau, dim=2).mean(dim=1, keepdim=True)


def gci3_pair_score(
    c: th.Tensor,
    off_c: th.Tensor,
    d: th.Tensor,
    off_d: th.Tensor,
    rel: th.Tensor,
    margin: float,
) -> th.Tensor:
    """Pairwise ELBE GCI3 overlap score :math:`\\|\\text{relu}(|c - r - d| - o_c - o_d + \\text{margin})\\|_2`.

    Preserves ELBE's actual overlap formula (:mod:`mowl.nn.el.elbe.losses` ``gci3_loss``)
    verbatim so that ``k = 1`` reproduces :class:`~mowl.nn.el.elbe.module.ELBEModule` exactly.
    """
    euc = th.abs(c - rel - d)
    return th.reshape(
        th.linalg.norm(th.relu(euc - off_c - off_d + margin), axis=1), [-1, 1]
    )


def gci3_loss(
    data: th.Tensor,
    class_embed: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    rel_embed: th.nn.Embedding,
    margin: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE GCI3 loss :math:`\\text{mean}_i \\text{agg}_j \\text{gci3\\_pair\\_score}(C_i, D_j, r)` for :math:`\\exists r.C \\sqsubseteq D`.

    Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci3_loss` at ``k = 1``.
    """
    _validate_aggregation(aggregation, tau)
    d_single = int(rel_embed.embedding_dim)
    r = rel_embed(data[:, 0])
    cen_c, off_c = _unpack_boxes(
        data[:, 1], class_embed, class_offset, n_components, embed_dim=d_single
    )
    cen_d, off_d = _unpack_boxes(
        data[:, 2], class_embed, class_offset, n_components, embed_dim=d_single
    )
    pair_scores = multi_pair_score(gci3_pair_score, cen_c, off_c, cen_d, off_d, r, margin)
    if aggregation == "hard":
        return th.amin(pair_scores, dim=2).mean(dim=1, keepdim=True)
    return softmin(pair_scores, tau=tau, dim=2).mean(dim=1, keepdim=True)


def gci3_bot_loss(
    data: th.Tensor,
    class_offset: th.nn.Embedding,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE GCI3-bot loss :math:`\\text{mean}_i \\|o_{C_i}\\|_2` for :math:`\\exists r.C \\sqsubseteq \\bot`.

    Returns shape ``(B,)``, matching :func:`mowl.nn.el.elbe.losses.gci3_bot_loss` at ``k = 1``.
    """
    _validate_aggregation(aggregation, tau)
    b_size = data.shape[0]
    k = _resolve_n_components(class_offset, n_components)
    col = 1 if data.shape[1] > 1 else 0
    raw_o = th.abs(class_offset(data[:, col]))
    d = raw_o.shape[-1] // k
    off_c = raw_o.view(b_size, k, d)
    return th.linalg.norm(off_c, dim=-1).mean(dim=1)


def class_assertion_loss(
    data: th.Tensor,
    ind_embed: th.nn.Embedding,
    ind_offset: th.nn.Embedding,
    class_embed: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    margin: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE class assertion loss :math:`\\text{agg}_m \\text{incl}(a, C_m)` for :math:`C(a)`.

    Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.class_assertion_loss` at ``k = 1``.
    """
    _validate_aggregation(aggregation, tau)
    cen_i = ind_embed(data[:, 0]).unsqueeze(1)  # (B, 1, d)
    off_i = th.abs(ind_offset(data[:, 0])).unsqueeze(1)  # (B, 1, d)
    d_single = int(ind_embed.embedding_dim)
    cen_c, off_c = _unpack_boxes(
        data[:, 1], class_embed, class_offset, n_components, embed_dim=d_single
    )
    incl = multi_inclusion(cen_i, off_i, cen_c, off_c, margin).squeeze(1)  # (B, k)
    if aggregation == "hard":
        return th.amin(incl, dim=1, keepdim=True)
    return softmin(incl, tau=tau, dim=1, keepdim=True)


def object_property_assertion_loss(
    data: th.Tensor,
    ind_embed: th.nn.Embedding,
    ind_offset: th.nn.Embedding,
    rel_embed: th.nn.Embedding,
    margin: float,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """ELBE object property assertion loss :math:`r(a, b)`, delegating to :mod:`mowl.nn.el.elbe.losses`."""
    _validate_aggregation(aggregation, tau)
    return EL.object_property_assertion_loss(
        data,
        ind_embed,
        ind_offset,
        rel_embed,
        margin,
        neg=neg,
    )
