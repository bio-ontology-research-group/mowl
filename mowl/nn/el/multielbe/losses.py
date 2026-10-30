"""Loss functions for MultiELBE.

Every concept name :math:`C` is represented as a union of :math:`k` axis-parallel
boxes :math:`C = \\bigcup_{i=1}^k C_i` with :math:`C_i = (c_{C,i}, |o_{C,i}|) \\subset \\mathbb{R}^d`.
Roles :math:`r` are represented by single translation vectors :math:`r \\in \\mathbb{R}^d`
as in ELBE, and individuals :math:`a` are represented by single boxes as in ELBE.

Guarantees:
- **Exact k = 1 identity with ELBE (identical to ELBEModule at equal margin):**
  When ``n_components == 1``, every loss function in this module (``gci0_loss``,
  ``gci0_bot_loss``, ``gci1_loss``, ``gci1_bot_loss``, ``gci2_loss`` positive and
  negative, ``gci3_loss``, ``gci3_bot_loss``, ``class_assertion_loss``,
  ``object_property_assertion_loss``) reduces algebraically and numerically (to
  floating-point tolerance ``1e-6``) to the corresponding function in
  :mod:`mowl.nn.el.elbe.losses` at equal ``margin``, including on ``gci1`` batches
  whose pairwise intersections are empty and preserving ELBE's margin sign
  convention (positive ``margin`` tightens).
- **Zero-loss characterisation:**
  - At ``margin = 0`` under ``aggregation="hard"``, ``gci0_loss`` is zero if and
    only if each component :math:`C_i` is contained in at least one component
    :math:`D_j`.
  - At ``margin = 0`` under ``aggregation="hard"``, ``gci1_loss`` is zero when at
    least one pair :math:`(C_i, D_j)` has a non-empty intersection
    (:math:`\\text{lo}_{ij} \\le \\text{up}_{ij}`) and every pair with non-empty
    intersection is contained in some component :math:`E_m`; disjoint component
    pairs :math:`(C_i, D_j)` are gated out and impose no inclusion penalty. When
    all :math:`k^2` pairs are disjoint, only the least-empty pair
    :math:`(i^*, j^*) = \\arg\\min_{i,j} \\text{empt}(C_i, D_j)` is gated in and the
    loss equals :math:`\\min_m \\text{incl}(I(C_{i^*}, D_{j^*}), E_m) + \\text{empt}(C_{i^*}, D_{j^*})`.
  - ``gci1_bot_loss`` is the mean of non-negative ELBE pairwise disjointness
    scores over all :math:`k^2` pairs :math:`(C_i, D_j)`, and is zero if and only
    if **all** :math:`k^2` pairs have zero ELBE disjointness score (i.e.
    :math:`|c_{C,i} - c_{D,j}| \\ge o_{C,i} + o_{D,j} + \\text{margin}` in every
    coordinate for every pair :math:`(i, j)`).
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
- **Non-zero soft loss at hard models and component collapse pressure:** Under
  ``aggregation="soft"`` at finite :math:`\\tau > 0`, :func:`~mowl.nn.el.softagg.softmin`
  satisfies :math:`\\min_j v_j \\le \\text{softmin}_\\tau(v) \\le \\text{mean}_j v_j`
  and is strictly positive whenever the target components do not all have zero
  inclusion score (e.g. :math:`\\text{softmin}_{\\tau=1}([0, 5]) \\approx 0.69`).
  Thus a hard zero-loss model is not a zero of the soft loss: every target
  component receives a gradient proportional to its softmax weight pulling it
  toward the source component, which promotes component collapse unless paired
  with :func:`~mowl.nn.el.softagg.diversity_penalty` (``diversity_reg > 0``).
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
from mowl.nn.el.softagg import soft_gci1_terms, softmin


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

    Guarantees:
    - Applies ``torch.abs`` to ``off_a`` and ``off_b`` before invoking ``fn``.
    - For ``ka = kb = 1``, ``multi_pair_score(fn, cen_a, off_a, cen_b, off_b, ...)[:, 0, 0]``
      equals ``fn(cen_a[:, 0], |off_a[:, 0]|, cen_b[:, 0], |off_b[:, 0]|, ...).reshape(-1)``.

    Non-guarantees:
    - Extra tensor arguments whose shape is not ``(B, d)`` are passed through
      without component-grid broadcasting.

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

    Non-guarantees:
    - When ``margin < 0``, the score can be zero even when ``A_{b,i}`` slightly
      exceeds ``B_{b,j}``.

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
    - At ``margin = 0`` and ``aggregation="hard"``, equals ``0`` if and only if
      every component :math:`C_i` is contained in at least one component :math:`D_j`.
    - Converges from ``aggregation="soft"`` to ``aggregation="hard"`` as ``tau -> 0+``.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    - Different source components :math:`C_i` may map into the same target
      component :math:`D_j`.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the loss is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`D_j` toward each :math:`C_i`, which is
      why ``diversity_reg`` exists).

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 2)`` with columns ``(C, D)``.
    class_embed : torch.nn.Embedding
        Class centre embedding table of width ``k * d``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    margin : float
        ELBE inclusion margin (positive margin tightens).
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    neg : bool, default False
        Unused flag kept for GCI dispatch signature uniformity.
    aggregation : str, default "hard"
        Target-component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-minimum temperature used when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row loss tensor of shape ``(B, 1)``.
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

    Guarantees:
    - Returns shape ``(B,)``, matching :func:`mowl.nn.el.elbe.losses.gci0_bot_loss` at ``k = 1``.
    - Equals ``0`` if and only if all :math:`k` component offsets of :math:`C` are zero.

    Non-guarantees:
    - Constrains half-widths to zero but does not constrain component centres.
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, >=1)`` whose column ``0`` holds class indices ``C``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    neg : bool, default False
        Unused flag kept for GCI dispatch signature uniformity.
    aggregation : str, default "hard"
        Aggregation mode validated for API consistency (``"hard"`` or ``"soft"``).
    tau : float, default 1.0
        Temperature validated when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row loss tensor of shape ``(B,)``.
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
    - Under ``aggregation="soft"`` (delegating to :func:`~mowl.nn.el.softagg.soft_gci1_terms`):
      :math:`w_{ij} = \\text{softmax}(-\\text{empt}_{ij} / \\tau)` (with gradient),
      emptiness term :math:`\\sum_{i,j} w_{ij} \\text{empt}_{ij}`, detached gate
      :math:`g_{ij} = \\max(\\text{non\\_empty}_{ij}^\\text{det}, w_{ij}^\\text{det})`
      (detached because the gate selects active pairs rather than acting as a
      target), and inner target aggregation :math:`\\text{softmin}_\\tau` over :math:`E_m`.

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci1_loss` at ``k = 1``
      on both empty and non-empty intersections under both ``"hard"`` and ``"soft"``.
    - At ``margin = 0`` under ``aggregation="hard"``:
      - Equals ``0`` when at least one pair :math:`(C_i, D_j)` has a non-empty
        intersection and every non-empty pair's intersection is contained in
        some component :math:`E_m` (disjoint pairs receive :math:`g_{ij} = 0`).
      - Is strictly positive when any non-empty pair's intersection lies outside
        every :math:`E_m`.
      - When all :math:`k^2` pairs are disjoint, only the least-empty pair
        :math:`(i^*, j^*)` is gated in and the loss equals
        :math:`\\min_m \\text{incl}(I(C_{i^*}, D_{j^*}), E_m) + \\text{empt}(C_{i^*}, D_{j^*})`.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    - If a non-empty intersection :math:`C_i \\cap D_j` is covered only by a
      union of multiple :math:`E_m` rather than a single :math:`E_m`, the loss
      remains positive.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the loss is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`E_m` toward each active intersection,
      which is why ``diversity_reg`` exists).
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

    non_empty = (lower.detach() <= upper.detach()).all(dim=-1)  # (B, k, k)
    empt_soft, gated_incl = soft_gci1_terms(empt, non_empty, incl_ijm, tau=tau)
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

    Computes the arithmetic mean of non-negative ELBE pairwise disjointness
    scores over all :math:`k^2` component pairs :math:`(C_i, D_j)` (differing
    from single-box :func:`mowl.nn.el.elbe.losses.gci1_bot_loss` only in this
    mean aggregation over the :math:`k^2` pairs, while retaining ELBE's
    Euclidean norm over coordinates inside each pair).

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci1_bot_loss` at ``k = 1``.
    - Because the loss is the mean of non-negative pairwise terms, it equals
      ``0`` if and only if **all** :math:`k^2` component pairs :math:`(C_i, D_j)`
      score ``0``—which under ELBE's :math:`L_2`-norm coordinate score means each
      pair is separated by at least ``margin`` in *every* coordinate (i.e.
      :math:`|c_{C,i} - c_{D,j}| \\ge o_{C,i} + o_{D,j} + \\text{margin}` in every
      coordinate for every :math:`(i, j)`).

    Non-guarantees:
    - Disjoint component pairs separated along one coordinate whose projections
      still satisfy :math:`|c_{C,i,m} - c_{D,j,m}| < o_{C,i,m} + o_{D,j,m} + \\text{margin}`
      along another coordinate :math:`m` still incur a positive score.
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, >=2)`` with columns ``(C, D, ...)``.
    class_embed : torch.nn.Embedding
        Class centre embedding table of width ``k * d``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    margin : float
        ELBE disjointness margin (positive margin requires larger separation).
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    neg : bool, default False
        Unused flag kept for GCI dispatch signature uniformity.
    aggregation : str, default "hard"
        Aggregation mode validated for API consistency (``"hard"`` or ``"soft"``).
    tau : float, default 1.0
        Temperature validated when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row loss tensor of shape ``(B, 1)``.
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
    """Positive ELBE GCI2 score :math:`\\text{mean}_i \\text{agg}_j \\text{gci2\\_score}(C_i, D_j, r)` for :math:`C \\sqsubseteq \\exists r.D`.

    Delegates to :func:`gci2_loss` with ``neg=False``.

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci2_score` at ``k = 1``.
    - At ``margin = 0`` and ``aggregation="hard"``, equals ``0`` if and only if
      every translated component :math:`C_i + r` is contained in at least one
      target component :math:`D_j`.

    Non-guarantees:
    - Roles are single translation vectors :math:`r \\in \\mathbb{R}^d` shared
      across all components.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the score is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`D_j` toward each :math:`C_i + r`, which
      is why ``diversity_reg`` exists).

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(C, r, D)``.
    class_embed : torch.nn.Embedding
        Class centre embedding table of width ``k * d``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    rel_embed : torch.nn.Embedding
        Relation translation embedding table of width ``d``.
    margin : float
        ELBE inclusion margin (positive margin tightens).
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    aggregation : str, default "hard"
        Target-component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-minimum temperature used when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row score tensor of shape ``(B, 1)``.
    """
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
    """ELBE GCI2 loss for :math:`C \\sqsubseteq \\exists r.D` (positive and negative).

    Aggregation for :math:`k \\ge 2`:
    - When ``neg=False``, :math:`s` is :func:`mowl.nn.el.elbe.losses.gci2_score`
      (:math:`\\|\\text{relu}(|c_{C,i} + r - c_{D,j}| + o_{C,i} - o_{D,j} + \\text{margin})\\|_2`),
      aggregated as :math:`\\text{mean}_i \\min_j s(C_i, D_j, r)` under ``"hard"``
      (or :math:`\\text{mean}_i \\text{softmin}_{\\tau, j} s(C_i, D_j, r)` under
      ``"soft"``), so every translated component :math:`C_i + r` must be
      contained in some :math:`D_j`.
    - When ``neg=True``, :math:`s` is :func:`mowl.nn.el.elbe.losses.gci2_score_neg`
      (:math:`\\|\\text{relu}(|c_{C,i} + r - c_{D,j}| - o_{C,i} - o_{D,j} - \\text{margin})\\|_2`),
      aggregated as the mean over all :math:`k^2` component pairs
      :math:`\\text{mean}_{i,j} s(C_i, D_j, r)` identically under both ``"hard"``
      and ``"soft"`` (no ``softmin``).

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci2_loss`
      at ``k = 1`` for both ``neg=False`` and ``neg=True``.
    - Converges from ``aggregation="soft"`` to ``aggregation="hard"`` as ``tau -> 0+``
      (and is identical for ``neg=True`` at all ``tau``).

    Non-guarantees:
    - Under ``neg=True``, returns the raw negative score :math:`\\text{mean}_{i,j} \\text{gci2\\_score\\_neg}(C_i, D_j, r)`
      (as ELBE does); pushing this score toward ``1`` is performed by the
      model's MSE criterion rather than inside ``gci2_loss``.
    - Under ``neg=False`` and ``aggregation="soft"`` at finite ``tau > 0``, the
      loss is not zero at a hard model (:func:`~mowl.nn.el.softagg.softmin`
      :math:`\\le \\text{mean}` pulls all target components :math:`D_j` toward each
      :math:`C_i + r`, which is why ``diversity_reg`` exists).

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(C, r, D)``.
    class_embed : torch.nn.Embedding
        Class centre embedding table of width ``k * d``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    rel_embed : torch.nn.Embedding
        Relation translation embedding table of width ``d``.
    margin : float
        ELBE margin (positive margin tightens inclusion and negative separation).
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    neg : bool, default False
        If ``False``, compute the positive inclusion score; if ``True``, compute
        the negative separation score.
    aggregation : str, default "hard"
        Target-component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-minimum temperature used when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row score tensor of shape ``(B, 1)``.
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
    if neg:
        pair_scores = multi_pair_score(
            EL.gci2_score_neg, cen_c, off_c, cen_d, off_d, r, margin
        )
        return pair_scores.mean(dim=(1, 2)).unsqueeze(1)
    pair_scores = multi_pair_score(
        EL.gci2_score, cen_c, off_c, cen_d, off_d, r, margin
    )
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
    verbatim so that ``k = 1`` reproduces :class:`~mowl.nn.el.elbe.module.ELBEModule`
    at equal ``margin``.

    Guarantees:
    - Returns shape ``(N, 1)`` matching the body of :func:`mowl.nn.el.elbe.losses.gci3_loss`.
    - At ``margin = 0``, equals ``0`` if and only if the shifted box
      :math:`(c - r, o_c)` overlaps :math:`(d, o_d)` in every coordinate.

    Non-guarantees:
    - Enforces box overlap (:math:`-o_c - o_d`) rather than full box inclusion
      (:math:`+o_c - o_d`), matching ELBE's implementation.

    Parameters
    ----------
    c : torch.Tensor
        Source box centres of shape ``(N, d)``.
    off_c : torch.Tensor
        Non-negative source box half-widths of shape ``(N, d)``.
    d : torch.Tensor
        Target box centres of shape ``(N, d)``.
    off_d : torch.Tensor
        Non-negative target box half-widths of shape ``(N, d)``.
    rel : torch.Tensor
        Relation translation vectors of shape ``(N, d)``.
    margin : float
        ELBE overlap margin (positive margin requires deeper overlap).

    Returns
    -------
    torch.Tensor
        Pairwise scores of shape ``(N, 1)``.
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

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.gci3_loss` at ``k = 1``.
    - At ``margin = 0`` and ``aggregation="hard"``, equals ``0`` if and only if
      each shifted component :math:`C_i - r` overlaps at least one target
      component :math:`D_j`.

    Non-guarantees:
    - Preserves ELBE's overlap formula rather than full box inclusion.
    - ``neg=True`` is not distinct from ``neg=False``.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the loss is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`D_j` toward each :math:`C_i - r`, which
      is why ``diversity_reg`` exists).

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(r, C, D)``.
    class_embed : torch.nn.Embedding
        Class centre embedding table of width ``k * d``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    rel_embed : torch.nn.Embedding
        Relation translation embedding table of width ``d``.
    margin : float
        ELBE overlap margin (positive margin tightens).
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    neg : bool, default False
        Unused flag kept for GCI dispatch signature uniformity.
    aggregation : str, default "hard"
        Target-component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-minimum temperature used when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row loss tensor of shape ``(B, 1)``.
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

    Indexes ``data[:, 1]`` for :math:`C` (rows are ``(r, C, bot)``) exactly as
    :func:`mowl.nn.el.elbe.losses.gci3_bot_loss` does.

    Guarantees:
    - Returns shape ``(B,)``, matching :func:`mowl.nn.el.elbe.losses.gci3_bot_loss` at ``k = 1``.
    - Equals ``0`` if and only if all :math:`k` component offsets of :math:`C` are zero.

    Non-guarantees:
    - Independent of the role :math:`r` in ``data[:, 0]`` (inherited from ELBE).
    - ``neg=True`` is not distinct from ``neg=False``.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, >=2)`` with columns ``(r, C, ...)``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    neg : bool, default False
        Unused flag kept for GCI dispatch signature uniformity.
    aggregation : str, default "hard"
        Aggregation mode validated for API consistency (``"hard"`` or ``"soft"``).
    tau : float, default 1.0
        Temperature validated when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row loss tensor of shape ``(B,)``.
    """
    _validate_aggregation(aggregation, tau)
    b_size = data.shape[0]
    k = _resolve_n_components(class_offset, n_components)
    raw_o = th.abs(class_offset(data[:, 1]))
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

    Guarantees:
    - Returns shape ``(B, 1)``, matching :func:`mowl.nn.el.elbe.losses.class_assertion_loss` at ``k = 1``.
    - At ``margin = 0`` and ``aggregation="hard"``, equals ``0`` if and only if
      the individual box :math:`a` is contained in at least one component :math:`C_m`.

    Non-guarantees:
    - Individuals are single boxes rather than multi-component unions.
    - ``neg=True`` is not distinct from ``neg=False``.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the loss is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`C_m` toward :math:`a`, which is why
      ``diversity_reg`` exists).

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 2)`` with columns ``(a, C)``.
    ind_embed : torch.nn.Embedding
        Individual centre embedding table of width ``d``.
    ind_offset : torch.nn.Embedding
        Individual offset embedding table of width ``d``.
    class_embed : torch.nn.Embedding
        Class centre embedding table of width ``k * d``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    margin : float
        ELBE inclusion margin (positive margin tightens).
    n_components : Optional[int], default None
        Number of components :math:`k` per class.
    neg : bool, default False
        Unused flag kept for GCI dispatch signature uniformity.
    aggregation : str, default "hard"
        Target-component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-minimum temperature used when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row loss tensor of shape ``(B, 1)``.
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
    """ELBE object property assertion loss for :math:`r(a, b)`, delegating to :mod:`mowl.nn.el.elbe.losses`.

    Guarantees:
    - Returns shape ``(B, 1)``, identical to :func:`mowl.nn.el.elbe.losses.object_property_assertion_loss`
      for both ``neg=False`` and ``neg=True``.

    Non-guarantees:
    - Individuals and roles are single-component embeddings, so ``aggregation``
      and ``tau`` are validated for signature uniformity but do not alter the score.

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 3)`` with columns ``(a, r, b)``.
    ind_embed : torch.nn.Embedding
        Individual centre embedding table of width ``d``.
    ind_offset : torch.nn.Embedding
        Individual offset embedding table of width ``d``.
    rel_embed : torch.nn.Embedding
        Relation translation embedding table of width ``d``.
    margin : float
        ELBE margin.
    neg : bool, default False
        If ``False``, compute positive inclusion score; if ``True``, compute
        negative separation score.
    aggregation : str, default "hard"
        Aggregation mode validated for API consistency (``"hard"`` or ``"soft"``).
    tau : float, default 1.0
        Temperature validated when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row score tensor of shape ``(B, 1)``.
    """
    _validate_aggregation(aggregation, tau)
    return EL.object_property_assertion_loss(
        data,
        ind_embed,
        ind_offset,
        rel_embed,
        margin,
        neg=neg,
    )
