r"""Closed-form geometry of 2-D octagons and of products of 2-D octagons.

A 2-D octagon is the set

.. math::

    O(h) = \{p \in \mathbb{R}^2 : n_k \cdot p \le h_k,\ k = 0, \dots, 7\}

with the eight fixed facet normals :data:`DIRECTIONS`, at angles :math:`k \cdot 45^\circ`:

.. math::

    n_0 = (1, 0),\ n_1 = (1, 1),\ n_2 = (0, 1),\ n_3 = (-1, 1),\
    n_4 = (-1, 0),\ n_5 = (-1, -1),\ n_6 = (0, -1),\ n_7 = (1, -1).

Direction :math:`k + 4 \pmod 8` is opposite to direction :math:`k`. In the notation of
Charpenay & Schockaert (IJCAI 2024, Eq. 6), :math:`O(h)` is
:math:`\mathrm{Octa}(x^-, x^+, y^-, y^+, u^-, u^+, v^-, v^+)` with :math:`u = y - x`,
:math:`v = x + y` and :math:`h = (x^+, v^+, y^+, u^+, -x^-, -v^-, -y^-, -u^-)`
(:func:`to_paper_bounds`, :func:`from_paper_bounds`).

A *product of octagons* in :math:`\mathbb{R}^d` (``d`` even) is
:math:`O(h^{(1)}) \times \dots \times O(h^{(d/2)})`, one octagon on each consecutive
coordinate pair :math:`(x_{2i}, x_{2i+1})`. All functions below take bound tensors of shape
``(..., 8)`` (a single octagon) or ``(..., P, 8)`` (a product of ``P`` octagons) and work
on the last axis.

Tightening (Proposition 1 of Charpenay & Schockaert 2024). For every direction :math:`k`,

.. math::

    T(h)_k = \min\Big(h_k,\ \min_{(i, j, \alpha, \beta):\ n_k = \alpha n_i + \beta n_j,\
    \alpha, \beta > 0} \alpha h_i + \beta h_j\Big),

where the pairs :math:`(i, j)` are the three pairs of directions that bracket :math:`n_k`
with an angle below :math:`180^\circ` (:data:`TIGHTENING_RULES`). If :math:`O(h)` is
non-empty, :math:`T(h)_k` is the support function :math:`\max_{p \in O(h)} n_k \cdot p`
(by LP duality an optimal dual solution of this two-dimensional LP uses at most two
constraints), so :math:`O(T(h)) = O(h)` and :math:`T(T(h)) = T(h)`. :math:`O(h)` is empty
iff :math:`T(h)_k + T(h)_{k+4} < 0` for some :math:`k` (the paper's "if
:math:`x_1^- > x_1^+` ... then the octagon is empty"; conversely an infeasibility
certificate of three constraints yields such a :math:`k`). :func:`emptiness_slack` returns
:math:`\min_k (T(h)_k + T(h)_{k+4})`, the smallest width of the octagon over the four
facet directions, which is non-negative iff the octagon is non-empty, and zero iff it is
non-empty with empty interior.
"""
from __future__ import annotations

import itertools
from typing import List, Tuple

import torch as th

#: The eight facet normals of a 2-D octagon, shape ``(8, 2)``. Row ``k`` is at angle
#: ``k * 45`` degrees; row ``(k + 4) % 8`` is its opposite.
DIRECTIONS: th.Tensor = th.tensor(
    [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0],
     [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]])

#: Index of the opposite direction for each direction.
OPPOSITE: Tuple[int, ...] = tuple((k + 4) % 8 for k in range(8))


def _tightening_rules() -> List[Tuple[int, int, int, float, float]]:
    """Return all ``(k, i, j, alpha, beta)`` with ``n_k = alpha n_i + beta n_j``, alpha, beta > 0.

    ``i`` and ``j`` range over pairs of distinct, non-opposite directions different from
    ``k``. For the octagon directions there are exactly three such pairs per ``k``.
    """
    rules = []
    dirs = DIRECTIONS.double()
    for k in range(8):
        for i, j in itertools.combinations(range(8), 2):
            if k in (i, j) or j == OPPOSITE[i]:
                continue
            mat = th.stack([dirs[i], dirs[j]], dim=1)
            alpha, beta = th.linalg.solve(mat, dirs[k]).tolist()
            if alpha > 1e-9 and beta > 1e-9:
                rules.append((k, i, j, float(alpha), float(beta)))
    return rules


#: ``(k, i, j, alpha, beta)``: the bound on direction ``k`` implied by the bounds on
#: directions ``i`` and ``j`` (24 rules, three per direction).
TIGHTENING_RULES: Tuple[Tuple[int, int, int, float, float], ...] = tuple(_tightening_rules())

_RULE_I = th.tensor([[r[1] for r in TIGHTENING_RULES if r[0] == k] for k in range(8)])
_RULE_J = th.tensor([[r[2] for r in TIGHTENING_RULES if r[0] == k] for k in range(8)])
_RULE_A = th.tensor([[r[3] for r in TIGHTENING_RULES if r[0] == k] for k in range(8)])
_RULE_B = th.tensor([[r[4] for r in TIGHTENING_RULES if r[0] == k] for k in range(8)])


def directions_like(h: th.Tensor) -> th.Tensor:
    """Return :data:`DIRECTIONS` on the device and with the dtype of ``h``."""
    return DIRECTIONS.to(device=h.device, dtype=h.dtype)


def from_center_radius(center: th.Tensor, radius: th.Tensor) -> th.Tensor:
    """Bounds of the octagon(s) :math:`h_k = n_k \\cdot c + r_k`.

    With ``radius >= 0`` the centre lies in the octagon, so the octagon is non-empty.

    :param center: Centres, shape ``(..., 2)``.
    :param radius: Radii, shape ``(..., 8)``.
    :return: Bounds, shape ``(..., 8)``.
    """
    return center @ directions_like(center).T + radius


def from_box(center: th.Tensor, offset: th.Tensor) -> th.Tensor:
    """Tight bounds of the axis-parallel rectangle(s) ``[center - offset, center + offset]``.

    The diagonal bounds are the tight ones, :math:`n_k \\cdot c + o_x + o_y` for
    :math:`k` odd, so the result is a fixed point of :func:`tighten`.

    :param center: Centres, shape ``(..., 2)``.
    :param offset: Non-negative half-widths, shape ``(..., 2)``.
    :return: Bounds, shape ``(..., 8)``.
    """
    ox, oy = offset[..., 0], offset[..., 1]
    s = ox + oy
    radius = th.stack([ox, s, oy, s, ox, s, oy, s], dim=-1)
    return from_center_radius(center, radius)


def tighten(h: th.Tensor) -> th.Tensor:
    """Canonical (tight) bounds :math:`T(h)` (Proposition 1 of Charpenay & Schockaert 2024).

    For a non-empty octagon the result is its support function on the eight directions;
    the point set is unchanged and the operation is idempotent. For an empty octagon the
    result is only meaningful through :func:`emptiness_slack` (which is then negative).

    :param h: Bounds, shape ``(..., 8)``.
    :return: Tightened bounds, shape ``(..., 8)``.
    """
    ri = _RULE_I.to(h.device)
    rj = _RULE_J.to(h.device)
    ra = _RULE_A.to(device=h.device, dtype=h.dtype)
    rb = _RULE_B.to(device=h.device, dtype=h.dtype)
    implied = ra * h[..., ri] + rb * h[..., rj]  # (..., 8, 3)
    return th.minimum(h, implied.amin(dim=-1))


def widths(h: th.Tensor) -> th.Tensor:
    """Widths :math:`T(h)_k + T(h)_{k+4}` along the four facet directions, shape ``(..., 4)``."""
    t = tighten(h)
    return t[..., :4] + t[..., 4:]


def emptiness_slack(h: th.Tensor) -> th.Tensor:
    """:math:`\\min_k (T(h)_k + T(h)_{k+4})`, shape ``(...)``.

    Non-negative iff the octagon is non-empty; zero iff it is non-empty with empty
    interior (a segment or a point).
    """
    return widths(h).amin(dim=-1)


def is_empty(h: th.Tensor, tol: float = 0.0) -> th.Tensor:
    """``True`` where the octagon is empty (``emptiness_slack < -tol``), shape ``(...)``."""
    return emptiness_slack(h) < -tol


def product_emptiness_slack(h: th.Tensor) -> th.Tensor:
    """Smallest :func:`emptiness_slack` over the ``P`` factors of a product, shape ``(...)``.

    A product of octagons is empty iff one of its factors is.

    :param h: Bounds, shape ``(..., P, 8)``.
    """
    return emptiness_slack(h).amin(dim=-1)


def intersect(h1: th.Tensor, h2: th.Tensor) -> th.Tensor:
    """Tight bounds of the intersection: :math:`T(\\min(h_1, h_2))`.

    Works for single octagons and, factor by factor, for products of octagons.
    """
    return tighten(th.minimum(h1, h2))


def translate(h: th.Tensor, t: th.Tensor) -> th.Tensor:
    """Bounds of the translated octagon(s) :math:`O(h) + t`: :math:`h_k + n_k \\cdot t`.

    Translation preserves tightness.

    :param h: Bounds, shape ``(..., 8)``.
    :param t: Translations, shape ``(..., 2)``.
    """
    return h + t @ directions_like(h).T


def contains(h_outer: th.Tensor, h_inner: th.Tensor, tol: float = 1e-6) -> th.Tensor:
    """``True`` where :math:`O(h_{inner}) \\subseteq O(h_{outer})`.

    An empty inner octagon is contained in everything; otherwise the test is
    :math:`T(h_{inner}) \\le h_{outer}` on all eight directions (support-function criterion).
    For products (shape ``(..., P, 8)``) apply it per factor and combine with
    :func:`product_contains`.
    """
    inner_empty = is_empty(h_inner, tol=tol)
    inside = (tighten(h_inner) <= h_outer + tol).all(dim=-1)
    return inner_empty | inside


def product_contains(h_outer: th.Tensor, h_inner: th.Tensor, tol: float = 1e-6) -> th.Tensor:
    """``True`` where the product :math:`\\prod O(h_{inner}) \\subseteq \\prod O(h_{outer})`.

    :param h_outer: Bounds, shape ``(..., P, 8)``.
    :param h_inner: Bounds, shape ``(..., P, 8)``.
    """
    inner_empty = (emptiness_slack(h_inner) < -tol).any(dim=-1)
    inside = (tighten(h_inner) <= h_outer + tol).all(dim=-1).all(dim=-1)
    return inner_empty | inside


def disjoint(h1: th.Tensor, h2: th.Tensor, tol: float = 0.0) -> th.Tensor:
    """``True`` where two octagons are disjoint (as closed sets)."""
    return is_empty(th.minimum(h1, h2), tol=tol)


def membership(points: th.Tensor, h: th.Tensor, tol: float = 1e-6) -> th.Tensor:
    """``True`` where ``points`` lie in the octagon(s).

    :param points: Points, shape ``(..., 2)``.
    :param h: Bounds, broadcastable to ``(..., 8)``.
    """
    return ((points @ directions_like(points).T) <= h + tol).all(dim=-1)


def witness_point(h: th.Tensor) -> th.Tensor:
    """A point of each non-empty octagon, shape ``(..., 2)``.

    Takes the midpoint :math:`x^*` of the projection :math:`[-T(h)_4, T(h)_0]` onto the
    first axis (the projection of a non-empty convex polygon is exactly this interval when
    the bounds are tight), then the midpoint of the fibre :math:`\\{y : (x^*, y) \\in O\\}`.
    The result is meaningless for empty octagons.
    """
    t = tighten(h)
    x = (t[..., 0] - t[..., 4]) / 2.0
    y_hi = th.stack([t[..., 2], t[..., 1] - x, t[..., 3] + x], dim=-1).amin(dim=-1)
    y_lo = th.stack([-t[..., 6], -t[..., 5] - x, -t[..., 7] + x], dim=-1).amax(dim=-1)
    return th.stack([x, (y_lo + y_hi) / 2.0], dim=-1)


def to_paper_bounds(h: th.Tensor) -> th.Tensor:
    """Convert bounds to :math:`(x^-, x^+, y^-, y^+, u^-, u^+, v^-, v^+)`.

    This is the parametrisation of Eq. 6 of Charpenay & Schockaert (IJCAI 2024).
    """
    return th.stack([-h[..., 4], h[..., 0], -h[..., 6], h[..., 2],
                     -h[..., 7], h[..., 3], -h[..., 5], h[..., 1]], dim=-1)


def from_paper_bounds(b: th.Tensor) -> th.Tensor:
    """Inverse of :func:`to_paper_bounds`."""
    xm, xp, ym, yp, um, up, vm, vp = b.unbind(dim=-1)
    return th.stack([xp, vp, yp, up, -xm, -vm, -ym, -um], dim=-1)


def area(h: th.Tensor) -> th.Tensor:
    """Area of each non-empty octagon (zero for empty ones), shape ``(...)``.

    Computed from the eight (possibly repeated) vertices of the tight octagon with the
    shoelace formula. Used only for diagnostics and tests.
    """
    t = tighten(h)
    d = directions_like(h)
    verts = []
    for k in range(8):
        k2 = (k + 1) % 8
        mat = th.stack([d[k], d[k2]], dim=0)
        rhs = th.stack([t[..., k], t[..., k2]], dim=-1)
        verts.append(th.linalg.solve(mat, rhs.unsqueeze(-1)).squeeze(-1))
    v = th.stack(verts, dim=-2)  # (..., 8, 2)
    x, y = v[..., 0], v[..., 1]
    a = 0.5 * (x * th.roll(y, -1, dims=-1) - th.roll(x, -1, dims=-1) * y).sum(dim=-1)
    nonempty = emptiness_slack(h) >= 0
    return th.where(nonempty, a.abs(), th.zeros_like(a))


__all__ = [
    "DIRECTIONS", "OPPOSITE", "TIGHTENING_RULES", "from_center_radius", "from_box",
    "tighten", "widths", "emptiness_slack", "is_empty", "product_emptiness_slack",
    "intersect", "translate", "contains", "product_contains", "disjoint", "membership",
    "witness_point", "to_paper_bounds", "from_paper_bounds", "area",
]
