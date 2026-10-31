"""Loss functions for MultiBoxEL.

Every concept name :math:`C` is represented as a union of :math:`k` axis-parallel
boxes :math:`C = \\bigcup_{i=1}^k C_i` with :math:`C_i = (c_{C,i}, |o_{C,i}|) \\subset \\mathbb{R}^d`,
accompanied by a single translation bump vector :math:`b_C \\in \\mathbb{R}^d` shared across
components. Roles :math:`r` are represented by single head and tail boxes
:math:`H_r, T_r` as in Box\\ :math:`^2`\\ EL, and individuals :math:`a` are represented by
single boxes and bumps.

Guarantees:
- **Exact k = 1 identity with Box²EL:** When ``n_components == 1``, every loss
  function in this module (``gci0_loss``, ``gci0_bot_loss``, ``gci1_loss``,
  ``gci1_bot_loss``, ``gci2_loss`` positive and negative, ``gci3_loss``,
  ``gci3_bot_loss``, ``class_assertion_loss``, ``object_property_assertion_loss``)
  reduces algebraically and numerically (to floating-point tolerance ``1e-6``)
  to the corresponding function in :mod:`mowl.nn.el.boxsquaredel.losses`,
  including on ``gci1`` batches whose pairwise intersections are empty.
- **Zero-loss characterisation at margin 0:** At ``gamma = 0``:
  - ``gci0_loss`` is zero if and only if each component :math:`C_i` is contained
    in at least one component :math:`D_j`.
  - ``gci1_loss`` is zero when at least one pair :math:`(C_i, D_j)` has a
    non-empty intersection (:math:`\\text{lo}_{ij} \\le \\text{up}_{ij}`) and every
    pair with non-empty intersection is contained in some component :math:`E_m`;
    disjoint component pairs :math:`(C_i, D_j)` are gated out and impose no
    inclusion penalty.
  - ``gci1_bot_loss`` is zero if and only if all :math:`k^2` component pairs
    :math:`(C_i, D_j)` satisfy :math:`\\text{dist}(C_i, D_j) \\ge 0` (separated or
    touching) in *every* coordinate (inherited from Box²EL): zero loss implies
    disjointness (up to boundary contact), but disjointness does not imply zero
    loss (zero :math:`\\Rightarrow` disjoint, disjoint :math:`\\nRightarrow` zero).

Non-guarantees:
- **Disjointness vs. coordinatewise separation in ``gci1_bot_loss``:** Two boxes
  that are disjoint in :math:`\\mathbb{R}^d` because they are separated along one
  coordinate still incur a positive ``gci1_bot_loss`` if their projections
  overlap along any other coordinate.
- **Non-convexity across components:** Taking minima over target components
  :math:`\\min_j \\text{incl}(C_i, D_j)` and gating over pairwise intersections makes
  the objective piecewise smooth rather than convex; gradient descent is not
  guaranteed to reach a global minimum from arbitrary random initialisations.
- **Non-zero soft loss at hard models and component collapse pressure:** Under
  ``aggregation="soft"`` at finite :math:`\\tau > 0`, :func:`~mowl.nn.el.softagg.softmin`
  satisfies :math:`\\min_j v_j \\le \\text{softmin}_\\tau(v) \\le \\text{mean}_j v_j`
  and is strictly positive whenever the target components do not all have zero
  inclusion score (e.g. :math:`\\text{softmin}_{\\tau=1}([0, 5]) \\approx 0.69`).
  Thus a hard zero-loss model is not a zero of the soft loss: every target
  component receives a gradient proportional to its softmax weight pulling it
  toward the source component, which promotes component collapse unless paired
  with :func:`~mowl.nn.el.softagg.diversity_penalty` (``diversity_reg > 0``).

References
----------
G. Federico, M. Leemhuis and O. Kutz, "Introducing Multiboxes: Increasing the
Trustworthiness of Ontology Embeddings", JOWO 2026.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch as th

import mowl.nn.el.boxsquaredel.losses as B2L
from mowl.nn.el.softagg import _validate_tau, soft_gci1_terms, softmin


def _validate_aggregation(
    aggregation: str, tau: Union[float, int, th.Tensor] = 1.0
) -> None:
    """Validate ``aggregation`` mode and ``tau`` when ``aggregation == 'soft'``."""
    if aggregation not in ("hard", "soft"):
        raise ValueError(
            f"aggregation must be 'hard' or 'soft', got {aggregation!r}"
        )
    if aggregation == "soft":
        _validate_tau(tau)


def _resolve_n_components(
    emb: th.nn.Embedding,
    n_components: Optional[int] = None,
    embed_dim: Optional[int] = None,
) -> int:
    """Resolve the number of components ``k`` from explicit argument, embedding attribute, or exact ``embed_dim`` match."""
    if n_components is not None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError(f"n_components must be an int >= 1, got {n_components!r}")
        k = int(n_components)
    elif hasattr(emb, "n_components"):
        k = int(getattr(emb, "n_components"))
    else:
        known_d = embed_dim if embed_dim is not None else getattr(emb, "embed_dim", None)
        if known_d is not None and int(emb.embedding_dim) == int(known_d):
            k = 1
        else:
            raise ValueError(
                "Cannot resolve n_components: pass n_components explicitly, set "
                "emb.n_components on the embedding, or provide an embedding whose "
                "width equals embed_dim (k = 1)."
            )
    if emb.embedding_dim % k != 0:
        raise ValueError(
            f"Embedding width {emb.embedding_dim} is not divisible by n_components={k}."
        )
    if embed_dim is not None and emb.embedding_dim != k * int(embed_dim):
        raise ValueError(
            f"Embedding width {emb.embedding_dim} does not equal "
            f"n_components ({k}) * embed_dim ({embed_dim})."
        )
    return k


def _unpack_boxes(
    idx: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    n_components: Optional[int] = None,
    embed_dim: Optional[int] = None,
) -> Tuple[th.Tensor, th.Tensor]:
    """Look up and reshape ``(centers, |offsets|)`` to ``(B, k, d)``."""
    b_size = idx.shape[0]
    k = _resolve_n_components(class_center, n_components, embed_dim=embed_dim)
    raw_c = class_center(idx)
    raw_o = th.abs(class_offset(idx))
    d = raw_c.shape[-1] // k
    return raw_c.view(b_size, k, d), raw_o.view(b_size, k, d)


def multi_inclusion(
    cen_a: th.Tensor,
    off_a: th.Tensor,
    cen_b: th.Tensor,
    off_b: th.Tensor,
    gamma: float,
) -> th.Tensor:
    """Compute pairwise Box²EL inclusion scores between two component sets.

    Guarantees:
    - For ``ka = kb = 1``, ``multi_inclusion(cen_a, off_a, cen_b, off_b, gamma)[:, 0, 0]``
      equals :func:`mowl.nn.el.boxsquaredel.losses.inclusion_score` on the squeezed boxes.
    - At ``gamma = 0``, entry ``(b, i, j)`` is ``0`` if and only if box ``A_{b,i}``
      is contained in box ``B_{b,j}``.

    Non-guarantees:
    - When ``gamma < 0``, the score can be zero even when ``A_{b,i}`` slightly
      exceeds ``B_{b,j}``.

    Parameters
    ----------
    cen_a : torch.Tensor
        Component centres of shape ``(B, ka, d)``.
    off_a : torch.Tensor
        Non-negative component half-widths of shape ``(B, ka, d)``.
    cen_b : torch.Tensor
        Component centres of shape ``(B, kb, d)``.
    off_b : torch.Tensor
        Non-negative component half-widths of shape ``(B, kb, d)``.
    gamma : float
        Inclusion margin :math:`\\gamma`.

    Returns
    -------
    torch.Tensor
        Pairwise inclusion scores of shape ``(B, ka, kb)``.
    """
    ca = cen_a.unsqueeze(2)
    oa = th.abs(off_a).unsqueeze(2)
    cb = cen_b.unsqueeze(1)
    ob = th.abs(off_b).unsqueeze(1)
    dist = th.abs(ca - cb) - oa - ob
    return th.linalg.norm(th.relu(dist + 2 * oa - gamma), dim=-1)


def multi_intersection(
    cen_a: th.Tensor,
    off_a: th.Tensor,
    cen_b: th.Tensor,
    off_b: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor]:
    """Compute pairwise lower and upper intersection corners for two component sets.

    Guarantees:
    - Entry ``(b, i, j)`` has non-empty intersection if and only if
      ``(lower[b, i, j] <= upper[b, i, j]).all()``.
    - For ``ka = kb = 1``, matches the ``(lower, upper)`` corners of
      :func:`mowl.nn.el.boxsquaredel.losses.box_intersection`.

    Non-guarantees:
    - When a pair is disjoint in coordinate ``m``, ``lower[..., m] > upper[..., m]``
      (phantom box); callers must check ``lower <= upper`` to test non-emptiness.

    Parameters
    ----------
    cen_a : torch.Tensor
        Component centres of shape ``(B, ka, d)``.
    off_a : torch.Tensor
        Non-negative component half-widths of shape ``(B, ka, d)``.
    cen_b : torch.Tensor
        Component centres of shape ``(B, kb, d)``.
    off_b : torch.Tensor
        Non-negative component half-widths of shape ``(B, kb, d)``.

    Returns
    -------
    lower : torch.Tensor
        Pairwise lower corners of shape ``(B, ka, kb, d)``.
    upper : torch.Tensor
        Pairwise upper corners of shape ``(B, ka, kb, d)``.
    """
    ca = cen_a.unsqueeze(2)
    oa = th.abs(off_a).unsqueeze(2)
    cb = cen_b.unsqueeze(1)
    ob = th.abs(off_b).unsqueeze(1)
    lower = th.maximum(ca - oa, cb - ob)
    upper = th.minimum(ca + oa, cb + ob)
    return lower, upper


def gci0_score(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared GCI0 score :math:`\\text{mean}_i \\text{agg}_j \\text{incl}(C_i, D_j)` for :math:`C \\sqsubseteq D`.

    Guarantees:
    - Reduces to :func:`mowl.nn.el.boxsquaredel.losses.gci0_score` when ``k = 1``.
    - Converges from ``aggregation="soft"`` to ``aggregation="hard"`` as ``tau -> 0+``.

    Non-guarantees:
    - Does not require different components :math:`C_i` to map into distinct
      components :math:`D_j`.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the score is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`D_j` toward each :math:`C_i`, which is
      why ``diversity_reg`` / :func:`~mowl.nn.el.softagg.diversity_penalty` is needed).

    Parameters
    ----------
    data : torch.Tensor
        Integer tensor of shape ``(B, 2)`` with columns ``(C, D)``.
    class_center : torch.nn.Embedding
        Class centre embedding table of width ``k * d``.
    class_offset : torch.nn.Embedding
        Class offset embedding table of width ``k * d``.
    gamma : float
        Inclusion margin.
    n_components : Optional[int], default None
        Number of components :math:`k`.
    aggregation : str, default "hard"
        Target-component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-minimum temperature used when ``aggregation="soft"``.

    Returns
    -------
    torch.Tensor
        Per-row score tensor of shape ``(B,)``.
    """
    _validate_aggregation(aggregation, tau)
    cen_c, off_c = _unpack_boxes(data[:, 0], class_center, class_offset, n_components)
    cen_d, off_d = _unpack_boxes(data[:, 1], class_center, class_offset, n_components)
    incl = multi_inclusion(cen_c, off_c, cen_d, off_d, gamma)
    if aggregation == "hard":
        return th.amin(incl, dim=2).mean(dim=1)
    return softmin(incl, tau=tau, dim=2).mean(dim=1)


def gci0_loss(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Squared GCI0 loss for :math:`C \\sqsubseteq D`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci0_loss` at ``k = 1``.
    - At ``gamma = 0`` and ``aggregation="hard"``, equals ``0`` iff every
      component :math:`C_i` is contained in some component :math:`D_j`.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False`` (GCI0 is not in ``neg_capable_gcis``).
    """
    return gci0_score(
        data,
        class_center,
        class_offset,
        gamma,
        n_components=n_components,
        aggregation=aggregation,
        tau=tau,
    ).square()


def gci0_bot_score(
    data: th.Tensor,
    class_offset: th.nn.Embedding,
    n_components: Optional[int] = None,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared GCI0-bot score :math:`\\text{mean}_i \\|o_{C_i}\\|_2` for :math:`C \\sqsubseteq \\bot`.

    Guarantees:
    - Reduces to :func:`mowl.nn.el.boxsquaredel.losses.gci0_bot_score` at ``k = 1``.

    Non-guarantees:
    - Constrains half-widths to zero but does not constrain component centres.
    """
    _validate_aggregation(aggregation, tau)
    b_size = data.shape[0]
    k = _resolve_n_components(class_offset, n_components)
    raw_o = th.abs(class_offset(data[:, 0]))
    d = raw_o.shape[-1] // k
    off_c = raw_o.view(b_size, k, d)
    return th.linalg.norm(off_c, dim=-1).mean(dim=1)


def gci0_bot_loss(
    data: th.Tensor,
    class_offset: th.nn.Embedding,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Squared GCI0-bot loss for :math:`C \\sqsubseteq \\bot`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci0_bot_loss` at ``k = 1``.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    """
    return gci0_bot_score(
        data,
        class_offset,
        n_components=n_components,
        aggregation=aggregation,
        tau=tau,
    ).square()


def gci1_score(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared gated GCI1 score for :math:`C \\sqcap D \\sqsubseteq E`.

    For each batch row under ``aggregation="hard"``:
    - :math:`(i^*, j^*) = \\arg\\min_{i,j} \\text{empt}(C_i, D_j)` (chosen on detached
      emptiness penalties so no gradient flows through the discrete argmin).
    - Gate :math:`g_{ij} = 1` if :math:`\\text{lo}_{ij} \\le \\text{up}_{ij}` in every
      coordinate or :math:`(i, j) = (i^*, j^*)`, else :math:`0` (detached).
    - Score :math:`= \\text{empt}(C_{i^*}, D_{j^*}) + \\frac{\\sum_{i,j} g_{ij} \\min_m \\text{incl}(I(C_i, D_j), E_m)}{\\sum_{i,j} g_{ij}}`.

    Under ``aggregation="soft"`` (delegating to :func:`~mowl.nn.el.softagg.soft_gci1_terms`):
    - :math:`w_{ij} = \\text{softmax}(-\\text{empt}_{ij} / \\tau)` (with gradient),
      emptiness term :math:`\\sum_{i,j} w_{ij} \\text{empt}_{ij}`, detached gate
      :math:`g_{ij} = \\max(\\text{non\\_empty}_{ij}^\\text{det}, w_{ij}^\\text{det})`
      (detached because the gate selects active pairs rather than acting as a
      target), and inner target aggregation :math:`\\text{softmin}_\\tau` over :math:`E_m`.

    Guarantees:
    - At ``k = 1``, the single pair ``(0, 0)`` is always the designated pair
      (:math:`g_{00} = 1`), so this score matches
      :func:`mowl.nn.el.boxsquaredel.losses.gci1_score` on both empty and
      non-empty intersections under both ``"hard"`` and ``"soft"``.
    - When at least one pair :math:`(C_i, D_j)` intersects non-emptily, any
      other pair that is strictly disjoint (:math:`\\text{lo}_{ij} > \\text{up}_{ij}`
      in some coordinate) receives :math:`g_{ij} = 0` under ``"hard"`` and
      imposes no penalty.

    Non-guarantees:
    - If a non-empty intersection :math:`C_i \\cap D_j` cannot be covered by a
      single component :math:`E_m` (even though it could be covered by a union
      of several :math:`E_m`), :math:`\\min_m \\text{incl}(I(C_i, D_j), E_m)` remains positive.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the score is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`E_m` toward each active intersection,
      which is why ``diversity_reg`` exists).
    """
    _validate_aggregation(aggregation, tau)
    b_size = data.shape[0]
    cen_c, off_c = _unpack_boxes(data[:, 0], class_center, class_offset, n_components)
    cen_d, off_d = _unpack_boxes(data[:, 1], class_center, class_offset, n_components)
    cen_e, off_e = _unpack_boxes(data[:, 2], class_center, class_offset, n_components)
    k = cen_c.shape[1]

    lower, upper = multi_intersection(cen_c, off_c, cen_d, off_d)
    inter_cen = (lower + upper) / 2
    inter_off = th.abs(upper - lower) / 2

    empt = th.linalg.norm(th.relu(lower - upper), dim=-1)  # (B, k, k)
    empt_flat = empt.view(b_size, k * k)

    # Compute inclusion for all (i, j, m) as (B, k, k, k)
    ic = inter_cen.unsqueeze(3)  # (B, k, k, 1, d)
    io = inter_off.unsqueeze(3)  # (B, k, k, 1, d)
    ce = cen_e.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k, d)
    oe = off_e.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k, d)
    dist_ie = th.abs(ic - ce) - io - oe
    incl_ijm = th.linalg.norm(th.relu(dist_ie + 2 * io - gamma), dim=-1)  # (B, k, k, k)

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
        return gated_incl + empt_designated

    non_empty = (lower.detach() <= upper.detach()).all(dim=-1)  # (B, k, k)
    empt_soft, gated_incl = soft_gci1_terms(empt, non_empty, incl_ijm, tau=tau)
    return gated_incl + empt_soft


def gci1_loss(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Squared GCI1 loss for :math:`C \\sqcap D \\sqsubseteq E`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci1_loss` at ``k = 1`` on
      both empty and non-empty intersections.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    """
    return gci1_score(
        data,
        class_center,
        class_offset,
        gamma,
        n_components=n_components,
        aggregation=aggregation,
        tau=tau,
    ).square()


def gci1_bot_score(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared GCI1-bot score :math:`\\text{mean}_{i,j} \\|\\text{relu}(-\\text{dist}(C_i, D_j) - \\gamma)\\|_2` for :math:`C \\sqcap D \\sqsubseteq \\bot`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci1_bot_score` at ``k = 1``.
    - At ``gamma = 0``, equals ``0`` if and only if every component pair
      :math:`(C_i, D_j)` satisfies :math:`\\text{dist}(C_i, D_j) \\ge 0` (separated
      or touching) in *every* coordinate (inherited from Box²EL); thus zero
      score implies all :math:`k^2` pairs are disjoint (up to boundary contact),
      whereas disjointness does not imply zero score (zero :math:`\\Rightarrow`
      disjoint, disjoint :math:`\\nRightarrow` zero).

    Non-guarantees:
    - Disjoint component pairs separated in one coordinate whose projections
      overlap in another coordinate (:math:`\\text{dist} < 0` in that coordinate)
      still incur a positive score at ``gamma = 0``.
    - When two boxes touch on their boundary (:math:`\\text{dist} = 0` in one
      coordinate and :math:`\\ge 0` in all others), the score at ``gamma = 0`` is zero.
    """
    _validate_aggregation(aggregation, tau)
    cen_c, off_c = _unpack_boxes(data[:, 0], class_center, class_offset, n_components)
    cen_d, off_d = _unpack_boxes(data[:, 1], class_center, class_offset, n_components)
    dist = (
        th.abs(cen_c.unsqueeze(2) - cen_d.unsqueeze(1))
        - off_c.unsqueeze(2)
        - off_d.unsqueeze(1)
    )
    pair_score = th.linalg.norm(th.relu(-dist - gamma), dim=-1)
    return pair_score.mean(dim=(1, 2))


def gci1_bot_loss(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Squared GCI1-bot loss for :math:`C \\sqcap D \\sqsubseteq \\bot`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci1_bot_loss` at ``k = 1``.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    """
    return gci1_bot_score(
        data,
        class_center,
        class_offset,
        gamma,
        n_components=n_components,
        aggregation=aggregation,
        tau=tau,
    ).square()


def gci2_score(
    cen_c: th.Tensor,
    off_c: th.Tensor,
    bump_c: th.Tensor,
    cen_d: th.Tensor,
    off_d: th.Tensor,
    bump_d: th.Tensor,
    cen_head: th.Tensor,
    off_head: th.Tensor,
    cen_tail: th.Tensor,
    off_tail: th.Tensor,
    gamma: float,
    delta: float,
    aggregation: str = "hard",
    tau: float = 1.0,
    active_c: Optional[th.Tensor] = None,
    active_d: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Positive GCI2 loss :math:`(\\text{mean}_i \\text{incl}(C_i + b_D, H_r) + \\text{agg}_j \\text{incl}(D_j + b_C, T_r)) / 2` for :math:`C \\sqsubseteq \\exists r.D`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci2_score` at ``k = 1``.
    - When ``active_c`` or ``active_d`` is provided under ``aggregation="hard"``,
      inactive source slots are excluded from the mean (with denominator clamped
      to ``>= 1``) and inactive target slots are excluded from the minimum.
    - Raises :class:`ValueError` if ``active_c`` or ``active_d`` is provided with
      ``aggregation="soft"`` (structural mode forces ``"hard"`` aggregation).

    Non-guarantees:
    - When every target slot in ``active_d`` is masked out for a row (an
      all-empty, i.e. unsatisfiable, target :math:`D`), the target minimum term
      ``term_2`` returns ``0.0`` (making the target half of
      :math:`C \\sqsubseteq \\exists r.D` vacuously ``0``; when every source slot
      in ``active_c`` is masked out as well, the entire score is ``0.0``).
    - Shares a single class bump vector :math:`b_C, b_D` across all components of
      :math:`C` and :math:`D`.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the score is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`D_j + b_C` toward :math:`T_r`, which is
      why ``diversity_reg`` exists).
    """
    _validate_aggregation(aggregation, tau)
    if (active_c is not None or active_d is not None) and aggregation != "hard":
        raise ValueError(
            "gci2_score requires aggregation='hard' when active_c or active_d is provided "
            f"(structural forces hard), got aggregation={aggregation!r}."
        )
    bumped_c = cen_c + bump_d.unsqueeze(1)  # (B, kc, d)
    bumped_d = cen_d + bump_c.unsqueeze(1)  # (B, kd, d)

    incl_1 = multi_inclusion(
        bumped_c, off_c, cen_head.unsqueeze(1), off_head.unsqueeze(1), gamma
    ).squeeze(2)  # (B, kc)
    incl_2 = multi_inclusion(
        bumped_d, off_d, cen_tail.unsqueeze(1), off_tail.unsqueeze(1), gamma
    ).squeeze(2)  # (B, kd)

    if active_c is None and active_d is None:
        if aggregation == "hard":
            return (incl_1.mean(dim=1) + th.amin(incl_2, dim=1)) / 2
        return (incl_1.mean(dim=1) + softmin(incl_2, tau=tau, dim=1)) / 2

    if active_c is None:
        term_1 = incl_1.mean(dim=1)
    else:
        act_c = active_c.to(dtype=th.bool)
        mask_c = act_c.to(dtype=incl_1.dtype)
        term_1 = th.where(act_c, incl_1, th.zeros_like(incl_1)).sum(dim=1) / (
            mask_c.sum(dim=1).clamp_min(1.0)
        )

    if active_d is None:
        term_2 = (
            th.amin(incl_2, dim=1)
            if aggregation == "hard"
            else softmin(incl_2, tau=tau, dim=1)
        )
    else:
        act_d = active_d.to(dtype=th.bool)
        has_d = act_d.any(dim=1)
        incl_2_masked = incl_2.masked_fill(~act_d, float("inf"))
        min_2 = th.amin(incl_2_masked, dim=1)
        term_2 = th.where(has_d, min_2, th.zeros_like(min_2))

    return (term_1 + term_2) / 2


def gci2_score_neg(
    cen_c: th.Tensor,
    off_c: th.Tensor,
    bump_c: th.Tensor,
    cen_d: th.Tensor,
    off_d: th.Tensor,
    bump_d: th.Tensor,
    cen_head: th.Tensor,
    off_head: th.Tensor,
    cen_tail: th.Tensor,
    off_tail: th.Tensor,
    gamma: float,
    delta: float,
    aggregation: str = "hard",
    tau: float = 1.0,
    active_c: Optional[th.Tensor] = None,
    active_d: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Negative GCI2 loss :math:`\\frac{1}{k^2} \\sum_{i,j} \\left[(\\delta - \\text{md}(C_i + b_D, H_r))^2 + (\\delta - \\text{md}(D_j + b_C, T_r))^2\\right]`.

    Computes the mean over all :math:`k^2` component pairs :math:`(C_i, D_j)` of
    Box\\ :math:`^2`\\ EL's negative GCI2 score, identical for ``aggregation="hard"``
    and ``aggregation="soft"`` (no ``softmin``).

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci2_score_neg` at ``k = 1``.
    - At :math:`k \\ge 2`, equals ``0`` if and only if every component pair
      :math:`(C_i, D_j)` is separated at target distance :math:`\\delta` from the
      respective role boxes (:math:`\\text{md}(C_i + b_D, H_r) = \\delta` for all
      :math:`i` and :math:`\\text{md}(D_j + b_C, T_r) = \\delta` for all :math:`j`).
    - Identical under ``aggregation="hard"`` and ``aggregation="soft"``.

    Non-guarantees:
    - ``aggregation`` and ``tau`` are validated for API uniformity but do not
      alter the negative score.
    """
    _validate_aggregation(aggregation, tau)
    bumped_c = cen_c + bump_d.unsqueeze(1)  # (B, kc, d)
    bumped_d = cen_d + bump_c.unsqueeze(1)  # (B, kd, d)

    dist_1 = (
        th.abs(bumped_c - cen_head.unsqueeze(1))
        - th.abs(off_c)
        - th.abs(off_head).unsqueeze(1)
    )
    md_1 = th.linalg.norm(th.relu(dist_1 + gamma), dim=-1)  # (B, kc)

    dist_2 = (
        th.abs(bumped_d - cen_tail.unsqueeze(1))
        - th.abs(off_d)
        - th.abs(off_tail).unsqueeze(1)
    )
    md_2 = th.linalg.norm(th.relu(dist_2 + gamma), dim=-1)  # (B, kd)

    sq_1 = (delta - md_1).square()
    sq_2 = (delta - md_2).square()

    if active_c is None:
        first_part = sq_1.mean(dim=1)
    else:
        act_c = active_c.to(dtype=th.bool)
        mask_c = act_c.to(dtype=sq_1.dtype)
        first_part = th.where(act_c, sq_1, th.zeros_like(sq_1)).sum(dim=1) / (
            mask_c.sum(dim=1).clamp_min(1.0)
        )

    if active_d is None:
        second_part = sq_2.mean(dim=1)
    else:
        act_d = active_d.to(dtype=th.bool)
        mask_d = act_d.to(dtype=sq_2.dtype)
        second_part = th.where(act_d, sq_2, th.zeros_like(sq_2)).sum(dim=1) / (
            mask_d.sum(dim=1).clamp_min(1.0)
        )

    return first_part + second_part


def gci2_loss(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    head_center: th.nn.Embedding,
    head_offset: th.nn.Embedding,
    tail_center: th.nn.Embedding,
    tail_offset: th.nn.Embedding,
    bump: th.nn.Embedding,
    gamma: float,
    delta: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """GCI2 loss for :math:`C \\sqsubseteq \\exists r.D` (positive and negative).

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci2_loss` at ``k = 1`` for
      both ``neg=False`` and ``neg=True``.

    Non-guarantees:
    - Roles remain single head/tail boxes rather than multi-box relations.
    """
    _validate_aggregation(aggregation, tau)
    d_single = int(bump.embedding_dim)
    cen_c, off_c = _unpack_boxes(
        data[:, 0], class_center, class_offset, n_components, embed_dim=d_single
    )
    cen_d, off_d = _unpack_boxes(
        data[:, 2], class_center, class_offset, n_components, embed_dim=d_single
    )
    bump_c = bump(data[:, 0])
    bump_d = bump(data[:, 2])

    cen_head = head_center(data[:, 1])
    off_head = th.abs(head_offset(data[:, 1]))
    cen_tail = tail_center(data[:, 1])
    off_tail = th.abs(tail_offset(data[:, 1]))

    fn = gci2_score_neg if neg else gci2_score
    return fn(
        cen_c,
        off_c,
        bump_c,
        cen_d,
        off_d,
        bump_d,
        cen_head,
        off_head,
        cen_tail,
        off_tail,
        gamma,
        delta,
        aggregation=aggregation,
        tau=tau,
    )


def gci3_score_boxes(
    cen_d: th.Tensor,
    off_d: th.Tensor,
    active: Optional[th.Tensor],
    cen_head: th.Tensor,
    off_head: th.Tensor,
    bump_c: th.Tensor,
    gamma: float = 0.0,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared GCI3 score :math:`\\text{agg}_m \\text{incl}(H_r - b_C, D_m)` on pre-unpacked boxes.

    Guarantees:
    - When ``active is None``, returns the exact same tensor as :func:`gci3_score`.
    - When ``active`` is provided under ``aggregation="hard"``, inactive target
      slots are excluded from the minimum.

    Non-guarantees:
    - When ``active`` is provided and every target slot in a row is masked out
      (an all-empty, i.e. unsatisfiable, target :math:`D`), returns ``0.0`` for
      that row, making :math:`\\exists r.C \\sqsubseteq D` vacuously ``0`` unless
      :math:`D` is forced non-empty by ``existence_loss``.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the score is not zero
      at a hard model.
    """
    _validate_aggregation(aggregation, tau)
    bumped_head_cen = (cen_head - bump_c).unsqueeze(1)  # (B, 1, d)
    bumped_head_off = off_head.unsqueeze(1)  # (B, 1, d)
    incl = multi_inclusion(
        bumped_head_cen, bumped_head_off, cen_d, off_d, gamma
    ).squeeze(1)
    if aggregation == "hard":
        if active is None:
            return th.amin(incl, dim=1)
        act = active.to(dtype=th.bool)
        has_act = act.any(dim=1)
        incl_masked = incl.masked_fill(~act, float("inf"))
        min_val = th.amin(incl_masked, dim=1)
        return th.where(has_act, min_val, th.zeros_like(min_val))
    return softmin(incl, tau=tau, dim=1)


def gci3_score(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    head_center: th.nn.Embedding,
    head_offset: th.nn.Embedding,
    tail_center: th.nn.Embedding,
    tail_offset: th.nn.Embedding,
    bump: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared GCI3 score :math:`\\text{agg}_m \\text{incl}(H_r - b_C, D_m)` for :math:`\\exists r.C \\sqsubseteq D`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci3_score` at ``k = 1``.

    Non-guarantees:
    - ``tail_center`` and ``tail_offset`` are accepted for signature compatibility
      with Box²EL and are not used in the score.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the score is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`D_m` toward :math:`H_r - b_C`, which is
      why ``diversity_reg`` exists).
    """
    _validate_aggregation(aggregation, tau)
    d_single = int(head_center.embedding_dim)
    cen_d, off_d = _unpack_boxes(
        data[:, 2], class_center, class_offset, n_components, embed_dim=d_single
    )
    cen_head = head_center(data[:, 0])
    off_head = th.abs(head_offset(data[:, 0]))
    bump_c = bump(data[:, 1])
    return gci3_score_boxes(
        cen_d,
        off_d,
        None,
        cen_head,
        off_head,
        bump_c,
        gamma=gamma,
        aggregation=aggregation,
        tau=tau,
    )


def gci3_loss(
    data: th.Tensor,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    head_center: th.nn.Embedding,
    head_offset: th.nn.Embedding,
    tail_center: th.nn.Embedding,
    tail_offset: th.nn.Embedding,
    bump: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Squared GCI3 loss for :math:`\\exists r.C \\sqsubseteq D`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.gci3_loss` at ``k = 1``.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    """
    return gci3_score(
        data,
        class_center,
        class_offset,
        head_center,
        head_offset,
        tail_center,
        tail_offset,
        bump,
        gamma,
        n_components=n_components,
        aggregation=aggregation,
        tau=tau,
    ).square()


def gci3_bot_score_boxes(
    off_head: th.Tensor,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared GCI3-bot score :math:`\\|o_{H_r}\\|_2` for :math:`\\exists r.C \\sqsubseteq \\bot` on box tensors.

    Guarantees:
    - Returns :math:`\\|\\text{abs}(\\text{off\\_head})\\|_2` of shape ``(B,)``,
      identical to :func:`gci3_bot_score`.

    Non-guarantees:
    - Independent of :math:`C` (follows Box²EL).
    """
    _validate_aggregation(aggregation, tau)
    return th.linalg.norm(th.abs(off_head), dim=-1)


def gci3_bot_score(
    data: th.Tensor,
    head_offset: th.nn.Embedding,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Unsquared GCI3-bot score :math:`\\|o_{H_r}\\|_2` for :math:`\\exists r.C \\sqsubseteq \\bot`.

    Guarantees:
    - Identical to :func:`mowl.nn.el.boxsquaredel.losses.gci3_bot_score`.

    Non-guarantees:
    - Independent of :math:`C`.
    """
    _validate_aggregation(aggregation, tau)
    off_head = th.abs(head_offset(data[:, 0]))
    return gci3_bot_score_boxes(
        off_head, aggregation=aggregation, tau=tau
    )


def gci3_bot_loss(
    data: th.Tensor,
    head_offset: th.nn.Embedding,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Squared GCI3-bot loss for :math:`\\exists r.C \\sqsubseteq \\bot`.

    Guarantees:
    - Identical to :func:`mowl.nn.el.boxsquaredel.losses.gci3_bot_loss`.

    Non-guarantees:
    - ``neg=True`` is not distinct from ``neg=False``.
    """
    return gci3_bot_score(
        data, head_offset, aggregation=aggregation, tau=tau
    ).square()


def class_assertion_score_boxes(
    cen_c: th.Tensor,
    off_c: th.Tensor,
    active: Optional[th.Tensor],
    cen_ind: th.Tensor,
    off_ind: th.Tensor,
    gamma: float = 0.0,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Class assertion score :math:`\\text{agg}_m \\text{incl}(a, C_m)` for :math:`C(a)` on box tensors.

    Guarantees:
    - When ``active is None``, returns the exact same tensor as
      :func:`class_assertion_loss`.
    - When ``active`` is provided under ``aggregation="hard"``, inactive target
      slots are excluded from the minimum.

    Non-guarantees:
    - When ``active`` is provided and every target slot in a row is masked out
      (an all-empty, i.e. unsatisfiable, class :math:`C`), returns ``0.0`` for
      that row, making :math:`C(a)` vacuously ``0`` unless :math:`C` is forced
      non-empty by ``existence_loss``.
    - Individuals remain single boxes rather than multi-boxes.
    """
    _validate_aggregation(aggregation, tau)
    cen_i = cen_ind.unsqueeze(1) if cen_ind.ndim == 2 else cen_ind
    off_i = th.abs(off_ind).unsqueeze(1) if off_ind.ndim == 2 else th.abs(off_ind)
    incl = multi_inclusion(cen_i, off_i, cen_c, off_c, gamma).squeeze(1)  # (B, K)
    if aggregation == "hard":
        if active is None:
            return th.amin(incl, dim=1)
        act = active.to(dtype=th.bool)
        has_act = act.any(dim=1)
        incl_masked = incl.masked_fill(~act, float("inf"))
        min_val = th.amin(incl_masked, dim=1)
        return th.where(has_act, min_val, th.zeros_like(min_val))
    return softmin(incl, tau=tau, dim=1)


def class_assertion_loss(
    data: th.Tensor,
    ind_center: th.nn.Embedding,
    ind_offset: th.nn.Embedding,
    class_center: th.nn.Embedding,
    class_offset: th.nn.Embedding,
    gamma: float,
    n_components: Optional[int] = None,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Class assertion loss :math:`\\text{agg}_m \\text{incl}(a, C_m)` for :math:`C(a)`.

    Guarantees:
    - Matches :func:`mowl.nn.el.boxsquaredel.losses.class_assertion_loss` at ``k = 1``.

    Non-guarantees:
    - Individuals remain single boxes rather than multi-boxes.
    - Under ``aggregation="soft"`` at finite ``tau > 0``, the score is not zero
      at a hard model (:func:`~mowl.nn.el.softagg.softmin` :math:`\\le \\text{mean}`
      pulls all target components :math:`C_m` toward :math:`a`, which is why
      ``diversity_reg`` exists).
    """
    _validate_aggregation(aggregation, tau)
    cen_i = ind_center(data[:, 0])
    off_i = th.abs(ind_offset(data[:, 0]))
    d_single = int(ind_center.embedding_dim)
    cen_c, off_c = _unpack_boxes(
        data[:, 1], class_center, class_offset, n_components, embed_dim=d_single
    )
    return class_assertion_score_boxes(
        cen_c,
        off_c,
        None,
        cen_i,
        off_i,
        gamma=gamma,
        aggregation=aggregation,
        tau=tau,
    )


def object_property_assertion_loss(
    data: th.Tensor,
    ind_center: th.nn.Embedding,
    ind_offset: th.nn.Embedding,
    head_center: th.nn.Embedding,
    head_offset: th.nn.Embedding,
    tail_center: th.nn.Embedding,
    tail_offset: th.nn.Embedding,
    bump_individuals: th.nn.Embedding,
    gamma: float,
    delta: float,
    neg: bool = False,
    aggregation: str = "hard",
    tau: float = 1.0,
) -> th.Tensor:
    """Object property assertion loss :math:`r(a, b)`, delegating to Box²EL unchanged.

    Guarantees:
    - Identical to :func:`mowl.nn.el.boxsquaredel.losses.object_property_assertion_loss`
      for both ``neg=False`` and ``neg=True`` and under both ``aggregation="hard"``
      and ``aggregation="soft"`` (individuals and roles have a single component).

    Non-guarantees:
    - Individuals and roles are single boxes.
    """
    _validate_aggregation(aggregation, tau)
    return B2L.object_property_assertion_loss(
        data,
        ind_center,
        ind_offset,
        head_center,
        head_offset,
        tail_center,
        tail_offset,
        bump_individuals,
        gamma,
        delta,
        neg=neg,
    )
