r"""Soft membership and quasi-Monte-Carlo volume for unions of boxes.

Building blocks of :class:`~mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule`, after
Aharoni, *Pushing the BoxEL Envelope* (MSc thesis, University of Amsterdam, 2024), Section 7.1.1.

A *multibox* is a union of ``n`` axis-parallel boxes. A box is stored as centre and
non-negative offset, :math:`B = (c - l, c + l)`. All functions here work on batches: ``centre``
and ``offset`` have shape ``(B, n, k)`` (``B`` multiboxes with ``n`` boxes each in
:math:`\mathbb{R}^k`) and the point set has shape ``(P, k)``.

Soft membership (thesis, Section 7.1.1). For one box with corners :math:`m, M`

.. math::
    \tilde\chi_B(p) = \mathrm{relu}\big(\mathrm{mean}(\sigma(M - p), \sigma(p - m)) - 0.5\big)

and for a multibox :math:`K = \bigcup_i B_i`, :math:`\tilde\chi_K(p) = \max_i \tilde\chi_{B_i}(p)`
(``relu`` is monotone, so this equals the thesis formula
:math:`\mathrm{relu}(\max_i \mathrm{mean}(\ldots) - 0.5)`). The thesis states that the
inner quantity exceeds 0.5 *iff* :math:`p \in K`. This is true if the aggregate over coordinates
is the minimum (``aggregate="min"``), but not if it is the mean: for a box with positive
offsets, :math:`\sigma(M_c - p_c) + \sigma(p_c - m_c) > 1` for every point and every coordinate,
so the mean is above 0.5 everywhere and :math:`\tilde\chi` is positive outside the box too
(it is only larger inside and decays outside). We therefore use ``aggregate="min"`` by default,
which realises the property the thesis states; ``aggregate="mean"`` is the literal formula.
The thesis uses no temperature; ``temperature`` divides the arguments of the sigmoid (``1.0`` is
the thesis).

Set operations on memberships (our choice, the thesis does not spell them out): intersection is
the pointwise minimum and union the pointwise maximum. The minimum of two multibox
memberships equals the maximum over all pairwise box intersections, so the intersection of
multiboxes is the union of pairwise intersections of their boxes.

Volume: :math:`\mathrm{Vol}(A) \approx \frac{(2 b_o)^k}{|P|}\sum_{p \in P}\tilde\chi_A(p)` for
a point set ``P`` in :math:`S = [-b_o, b_o]^k`.
"""

import torch as th
from torch.quasirandom import SobolEngine
from torch.utils.checkpoint import checkpoint

_AGGREGATES = ("mean", "min")


def sobol_points(num_points, dim, bound, seed=0, device="cpu"):
    """Deterministic scrambled Sobol point set in :math:`[-b_o, b_o]^{dim}`.

    The thesis only says that ``#QMC`` points are distributed uniformly over the space once at
    the start of training; the Sobol sequence (scrambled, seeded) is our choice of
    quasi-random sequence.

    :param num_points: Number of points.
    :type num_points: int
    :param dim: Dimension of the space.
    :type dim: int
    :param bound: Half-width :math:`b_o` of the cube.
    :type bound: float
    :param seed: Scrambling seed; the same seed always gives the same points.
    :type seed: int
    :return: Tensor of shape ``(num_points, dim)``.
    :rtype: :class:`torch.Tensor`
    """
    engine = SobolEngine(dimension=dim, scramble=True, seed=seed)
    unit = engine.draw(num_points).to(th.float32)
    return ((unit * 2.0 - 1.0) * bound).to(device)


def _aggregate(values, aggregate):
    """Aggregate over the last dimension."""
    if aggregate == "mean":
        return values.mean(dim=-1)
    if aggregate == "min":
        return values.amin(dim=-1)
    raise ValueError(f"aggregate must be one of {_AGGREGATES}, got {aggregate!r}.")


def soft_box_membership(points, centre, offset, temperature=1.0, aggregate="min"):
    """Soft membership of every point in every box (relu of the thesis' inner quantity).

    :param points: Points, shape ``(P, k)``.
    :param centre: Box centres, shape ``(B, n, k)``.
    :param offset: Non-negative box offsets, shape ``(B, n, k)``.
    :return: Tensor of shape ``(B, P, n)`` with values in ``[0, 0.5)``.
    """
    lo = (centre - offset).unsqueeze(1)
    hi = (centre + offset).unsqueeze(1)
    p = points.view(1, -1, 1, points.shape[-1])
    upper = th.sigmoid((hi - p) / temperature)
    lower = th.sigmoid((p - lo) / temperature)
    both = th.cat([upper, lower], dim=-1)
    return th.relu(_aggregate(both, aggregate) - 0.5)


def soft_membership(points, centre, offset, temperature=1.0, aggregate="min"):
    """Soft membership :math:`\\tilde\\chi_K(p)` in a multibox (maximum over its boxes).

    :return: Tensor of shape ``(B, P)``.
    """
    return soft_box_membership(points, centre, offset, temperature, aggregate).amax(dim=-1)


def hard_box_membership(points, centre, offset):
    """Exact (closed) membership of every point in every box; shape ``(B, P, n)``, bool."""
    lo = (centre - offset).unsqueeze(1)
    hi = (centre + offset).unsqueeze(1)
    p = points.view(1, -1, 1, points.shape[-1])
    return ((p >= lo) & (p <= hi)).all(dim=-1)


def hard_membership(points, centre, offset):
    """Exact membership in a multibox, shape ``(B, P)``, bool."""
    return hard_box_membership(points, centre, offset).any(dim=-1)


def overlap_slack(centre1, offset1, centre2, offset2, bound):
    """Per-coordinate overlap of two sets of boxes, clipped to :math:`S = [-b_o, b_o]^k`.

    :param centre1: ``(B, n1, k)``; :param offset1: ``(B, n1, k)``.
    :param centre2: ``(B, n2, k)``; :param offset2: ``(B, n2, k)``.
    :return: ``(B, n1, n2, k)``; entry ``[b, i, j, c]`` is ``min(M) - max(m)`` of box ``i`` of the
        first multibox, box ``j`` of the second and :math:`S` in coordinate ``c``. The three boxes
        have a common point iff all coordinates are non-negative.
    """
    lo1, hi1 = (centre1 - offset1).unsqueeze(2), (centre1 + offset1).unsqueeze(2)
    lo2, hi2 = (centre2 - offset2).unsqueeze(1), (centre2 + offset2).unsqueeze(1)
    lo = th.maximum(lo1, lo2).clamp(min=-bound)
    hi = th.minimum(hi1, hi2).clamp(max=bound)
    return hi - lo


def soft_overlap_gate(centre1, offset1, centre2, offset2, bound, temperature=1.0,
                      aggregate="min"):
    """Soft indicator that box ``i`` of one multibox meets box ``j`` of another inside :math:`S`.

    Same construction as :func:`soft_box_membership`, applied to the overlap slack instead of the
    point-to-face distance; it is positive iff the boxes overlap (``aggregate="min"``).

    :return: ``(B, n1, n2)``.
    """
    slack = overlap_slack(centre1, offset1, centre2, offset2, bound)
    return th.relu(_aggregate(th.sigmoid(slack / temperature), aggregate) - 0.5)


def hard_overlap(centre1, offset1, centre2, offset2, bound):
    """Exact version of :func:`soft_overlap_gate`; ``(B, n1, n2)``, bool."""
    return (overlap_slack(centre1, offset1, centre2, offset2, bound) > 0).all(dim=-1)


def project_gate(role_centre, role_offset, filler_centre, filler_offset, bound,
                 temperature=1.0, aggregate="min"):
    r"""Weight of each role box in the projection of a role restricted to a filler.

    The projection is :math:`\pi[\llbracket R\rrbracket \cap (S\times\llbracket D\rrbracket)]`.

    A role box is :math:`R_j = R^x_j \times R^y_j \subseteq \mathbb{R}^{2d}` (first ``d``
    coordinates: domain, last ``d``: codomain). The projection of
    :math:`R_j \cap (S \times D)` onto the domain is :math:`R^x_j \cap S` if
    :math:`R^y_j \cap D \cap S \neq \emptyset` and empty otherwise, so the projection of a
    union is a union of role-domain boxes and no sampling is needed for the existential
    quantifier. This function returns, for each ``j``, the soft indicator of the non-emptiness
    condition (maximum over the boxes of the filler).

    :param role_centre: ``(B, n, 2d)``; :param role_offset: ``(B, n, 2d)``.
    :param filler_centre: ``(B, m, d)``; :param filler_offset: ``(B, m, d)``.
    :return: ``(B, n)``.
    """
    d = filler_centre.shape[-1]
    gate = soft_overlap_gate(role_centre[..., d:], role_offset[..., d:],
                             filler_centre, filler_offset, bound, temperature, aggregate)
    return gate.amax(dim=-1)


def projection_membership(points, role_centre, role_offset, gate, temperature=1.0,
                          aggregate="min"):
    """Soft membership of points (in :math:`\\mathbb{R}^d`) in the domain projection of a role.

    :param gate: Output of :func:`project_gate`, ``(B, n)``.
    :return: ``(B, P)``: ``max_j min(chi_{R^x_j}(p), gate_j)``.
    """
    d = points.shape[-1]
    per_box = soft_box_membership(points, role_centre[..., :d], role_offset[..., :d],
                                  temperature, aggregate)
    return th.minimum(per_box, gate.unsqueeze(1)).amax(dim=-1)


def volume_scale(bound, dim):
    """Volume :math:`(2 b_o)^{dim}` of the cube :math:`S`."""
    return (2.0 * bound) ** dim


def qmc_volume(membership, bound, dim):
    """Quasi-Monte-Carlo volume from memberships over a fixed point set.

    :param membership: ``(..., P)`` memberships (soft or 0/1).
    :return: ``(...)`` volumes :math:`(2 b_o)^{dim} \\cdot \\mathrm{mean}_p\\,\\chi(p)`.
    """
    return volume_scale(bound, dim) * membership.mean(dim=-1)


def chunked_means(fn, points, chunk_size):
    """Mean over a point set of several membership tensors, computed chunk by chunk.

    ``fn`` maps a chunk of points to a tuple of tensors of shape ``(B, P_chunk)``. The result is a
    tuple of ``(B,)`` tensors: the means over *all* points. When gradients are needed each chunk
    is wrapped in activation checkpointing, so that peak memory is that of one chunk and not of
    the whole point set (at the price of a second forward pass of each chunk).

    :param fn: Callable ``points_chunk -> tuple of (B, P_chunk)``.
    :param points: ``(P, k)``.
    :param chunk_size: Points per chunk.
    :type chunk_size: int
    """
    total = points.shape[0]
    sums = None
    use_ckpt = th.is_grad_enabled()
    for start in range(0, total, chunk_size):
        chunk = points[start:start + chunk_size]

        def run(c):
            return th.stack([m.sum(dim=-1) for m in fn(c)])

        part = checkpoint(run, chunk, use_reentrant=False) if use_ckpt else run(chunk)
        sums = part if sums is None else sums + part
    return tuple(sums / total)


def containment_loss(numerator, denominator, eps=1e-8):
    r""":math:`1 - \mathrm{Vol}(A\cap B)/\mathrm{Vol}(A)` from the two (mean) memberships.

    The scale :math:`(2 b_o)^k` cancels. If :math:`\mathrm{Vol}(A) = 0` the loss is 1 (and has no
    gradient), as in the thesis' formula with the empty-set case made explicit.
    """
    return 1.0 - numerator / (denominator + eps)
