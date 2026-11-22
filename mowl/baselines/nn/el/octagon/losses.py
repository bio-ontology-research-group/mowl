r"""Losses for octagon-product embeddings of :math:`\mathcal{EL}^{++}` normal forms.

Every concept is a product of 2-D octagons, given by a bound tensor ``h`` of shape
``(B, P, 8)`` (``P = d / 2`` coordinate pairs, eight bounds per pair, see
:mod:`mowl.nn.el.octagon.geometry`). Roles are translations :math:`t_R \in \mathbb{R}^d`
and individuals are points.

Every loss below is *exact*: at the stated margin, it is zero if and only if the stated
geometric condition holds (for the conditions on concepts the sub-concept is assumed
non-empty, which the module guarantees for concept names by construction).

=================================  =========================================================
Function                           Zero iff
=================================  =========================================================
:func:`inclusion_loss`             :math:`T(h_C)_k + m \le h_{D,k}` for all factors and
                                   directions; at :math:`m = 0`: :math:`C \subseteq D`
:func:`nonempty_loss`              the product is non-empty
:func:`disjoint_loss`              :math:`\min_{p,k} \mathrm{width}_{p,k}(C \cap D) \le -m`;
                                   at :math:`m > 0`: :math:`C \cap D = \emptyset`; at
                                   :math:`m = 0`: :math:`C \cap D` has empty interior
:func:`interior_empty_loss`        the product has empty interior
:func:`point_inclusion_loss`       the point lies in the product (at :math:`m = 0`)
:func:`non_inclusion_loss`         some bound of :math:`C` exceeds the one of :math:`D` by
                                   at least :math:`m` (at :math:`m = 0`:
                                   :math:`C \not\subseteq D` up to the boundary)
=================================  =========================================================

Inclusion and point losses aggregate like ELBE: for each factor and each of the four
facet axes they take the larger violation of the two opposite directions,
:math:`\mathrm{relu}(\max(v_k, v_{k+4}))`, and return the Euclidean norm of all these
entries. On the two axis-parallel facet axes this is exactly ELBE's per-coordinate term
:math:`\mathrm{relu}(|c_C - c_D| + o_C - o_D + m)`; see :func:`inclusion_loss`.
"""
from __future__ import annotations

import torch as th

from mowl.nn.el.octagon import geometry as G


def _axis_max(v: th.Tensor) -> th.Tensor:
    """Fold ``(..., 8)`` direction values to ``(..., 4)``: the larger of opposite pairs."""
    return th.maximum(v[..., :4], v[..., 4:])


def _norm(v: th.Tensor) -> th.Tensor:
    """Euclidean norm over all axes but the first, shape ``(B,)``."""
    return th.linalg.norm(v.reshape(v.shape[0], -1), dim=-1)


def inclusion_loss(h_sub: th.Tensor, h_sup: th.Tensor, margin: float = 0.0) -> th.Tensor:
    r"""Loss of :math:`C \subseteq D` for octagon products.

    :math:`\big\| \mathrm{relu}\big(\max(v_k, v_{k+4})\big)_{p, k < 4} \big\|_2` with
    :math:`v = T(h_C) - h_D + m`.

    Zero iff :math:`T(h_C) + m \le h_D` on every factor and direction. For a non-empty
    :math:`C` and :math:`m = 0` this is exactly :math:`C \subseteq D` (support-function
    criterion: :math:`T(h_C)_k = \max_{p \in C} n_k \cdot p`).

    If both are boxes (:func:`~mowl.nn.el.octagon.geometry.from_box`), the four entries of a
    factor are :math:`(a, b, \delta_1, \delta_3)` where :math:`a, b` are ELBE's
    per-coordinate terms :math:`\mathrm{relu}(|c_C - c_D| + o_C - o_D + m)` of the two
    coordinates and :math:`\delta_1, \delta_3 \le a + b` are the diagonal terms. Hence
    ``ELBE <= octagon <= sqrt(5) * ELBE`` and both are zero on the same configurations.

    :param h_sub: Bounds of :math:`C`, shape ``(B, P, 8)``.
    :param h_sup: Bounds of :math:`D`, shape ``(B, P, 8)``.
    :param margin: Margin; positive values require strict inclusion, as in ELBE.
    :return: Shape ``(B,)``.
    """
    v = G.tighten(h_sub) - h_sup + margin
    return _norm(th.relu(_axis_max(v)))


def nonempty_loss(h: th.Tensor) -> th.Tensor:
    """Loss of non-emptiness: :math:`\\|\\mathrm{relu}(-\\mathrm{widths}(h))\\|_2`.

    Zero iff every factor is non-empty, i.e. iff the product is non-empty.

    :param h: Bounds, shape ``(B, P, 8)``.
    :return: Shape ``(B,)``.
    """
    return _norm(th.relu(-G.widths(h)))


def disjoint_loss(h1: th.Tensor, h2: th.Tensor, margin: float = 0.0) -> th.Tensor:
    r"""Loss of :math:`C \cap D = \emptyset`: :math:`\mathrm{relu}(s + m)`.

    :math:`s` is the smallest width of :math:`C \cap D` over all factors and facet axes
    (:func:`~mowl.nn.el.octagon.geometry.product_emptiness_slack` of
    :math:`\min(h_C, h_D)`). A product is empty iff one of its factors is, and a factor is
    empty iff one of its widths is negative, so the loss is zero iff :math:`s \le -m`:
    for :math:`m > 0` this implies :math:`C \cap D = \emptyset` (with a gap); for
    :math:`m = 0` it means :math:`C \cap D` is empty or has empty interior (touching
    regions count as disjoint). As for the prism model, the gradient flows through the
    single smallest width.

    :param h1: Bounds of :math:`C`, shape ``(B, P, 8)``.
    :param h2: Bounds of :math:`D`, shape ``(B, P, 8)``.
    :return: Shape ``(B,)``.
    """
    s = G.product_emptiness_slack(th.minimum(h1, h2))
    return th.relu(s + margin)


def interior_empty_loss(h: th.Tensor) -> th.Tensor:
    """Loss of "empty interior": :math:`\\mathrm{relu}(s)` with :math:`s` the smallest width.

    The module keeps concept names non-empty by construction, so :math:`C \\sqsubseteq
    \\bot` can only be approximated by flattening :math:`C`; the loss is zero iff the product
    has empty interior (measure zero).

    :param h: Bounds, shape ``(B, P, 8)``.
    :return: Shape ``(B,)``.
    """
    return th.relu(G.product_emptiness_slack(h))


def point_inclusion_loss(points: th.Tensor, h: th.Tensor, margin: float = 0.0) -> th.Tensor:
    """Loss of point membership, aggregated like :func:`inclusion_loss`.

    :param points: Points, shape ``(B, P, 2)``.
    :param h: Bounds, shape ``(B, P, 8)``.
    :return: Shape ``(B,)``; zero iff every :math:`n_k \\cdot p + m \\le h_k`.
    """
    v = points @ G.directions_like(h).T - h + margin
    return _norm(th.relu(_axis_max(v)))


def non_inclusion_loss(h_sub: th.Tensor, h_sup: th.Tensor, margin: float = 0.0) -> th.Tensor:
    r"""Loss of :math:`C \not\subseteq D`: :math:`\mathrm{relu}(m - \max_{p,k} v_{p,k})`.

    :math:`v = T(h_C) - h_D`. For non-empty :math:`C`, :math:`C \not\subseteq D` iff
    :math:`\max v > 0`, so the loss is zero iff :math:`C` sticks out of :math:`D` by at
    least :math:`m` in some facet direction. Used for negative samples of
    :math:`C \sqsubseteq \exists R.D`.

    :return: Shape ``(B,)``.
    """
    v = G.tighten(h_sub) - h_sup
    return th.relu(margin - v.reshape(v.shape[0], -1).amax(dim=-1))


def translation_loss(p_a: th.Tensor, t: th.Tensor, p_b: th.Tensor) -> th.Tensor:
    r"""Loss of :math:`R(a, b)` under translation semantics: :math:`\|p_a + t_R - p_b\|_2`.

    :param p_a: Points, shape ``(B, d)``.
    :param t: Translations, shape ``(B, d)``.
    :param p_b: Points, shape ``(B, d)``.
    :return: Shape ``(B,)``; zero iff :math:`p_a + t_R = p_b`.
    """
    return th.linalg.norm(p_a + t - p_b, dim=-1)
