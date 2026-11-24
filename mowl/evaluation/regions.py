"""Closed class regions of trained geometric :math:`\\mathcal{EL}` embeddings and exact
decision procedures for the propositional normal forms on them.

Every region family that a mOWL model can train implements the small :class:`Region`
protocol:

* :meth:`Region.is_empty` -- is the (closed) region the empty set;
* :meth:`Region.has_interior` -- does it contain a Euclidean ball of radius
  :math:`> \\mathrm{tol}/2` (with ``tol = 0``: is its topological interior non-empty);
* :meth:`Region.intersect` -- the intersection, as a region of the same family (or a
  :class:`BallIntersection` for balls, or a :class:`UnionRegion` for unions);
* :meth:`Region.contains` -- is ``other`` a subset of ``self``;
* :meth:`Region.contains_points` / :meth:`Region.bounding_box` -- pointwise membership,
  used to test the decision procedures against dense sampling.

With these, the normal forms :math:`C \\sqsubseteq D` (``gci0``),
:math:`C \\sqcap D \\sqsubseteq E` (``gci1``), :math:`C \\sqsubseteq \\bot` and
:math:`C \\sqcap D \\sqsubseteq \\bot` (``gci0_bot``, ``gci1_bot``) are decided by
:func:`satisfies`. All regions are **closed** sets: two regions that touch intersect.

Families and what is decided exactly:

* :class:`BoxRegion` -- closed axis-parallel boxes; intersection, emptiness and
  containment are coordinatewise (the semantics of
  :class:`mowl.geometric.checker.ModelChecker`, including its tolerance rules).
* :class:`BallRegion` / :class:`BallIntersection` -- closed Euclidean balls. Ball in
  ball: :math:`\\|c_1 - c_2\\| + r_1 \\le r_2`. Emptiness of an intersection of
  :math:`m` balls: exactly, by maximising the concave dual
  :math:`\\varphi(\\lambda) = \\sum_i \\lambda_i (\\|c_i\\|^2 - r_i^2)
  - \\|\\sum_i \\lambda_i c_i\\|^2`
  of :math:`\\min_x \\max_i (\\|x - c_i\\|^2 - r_i^2)` over the simplex by enumerating the
  supports of at most :math:`d + 1` affinely independent centres (the primal minimiser
  :math:`x^* = \\sum_i \\lambda_i c_i` is a witness point). Lens in ball
  (:math:`B_1 \\cap B_2 \\subseteq B_3`): closed form, the farthest point of the lens from
  the centre of :math:`B_3` lies on one of its two spherical caps and is the maximiser of a
  linear function over a spherical cap (:func:`lens_max_distance`). An intersection of
  three or more balls inside a ball is not decided (:class:`UndecidedInclusion`); it does
  not occur in the binary normal forms.
* :class:`PrismRegion` -- triangle :math:`\\times` box with the fixed triangle normals of
  :mod:`mowl.nn.el.prism.losses`. Every support vector :math:`h` with
  :math:`\\sum_k h_k \\ge 0` is the support function of its (equilateral) triangle, so
  prisms are closed under intersection (minimum of supports, intersection of boxes) and
  :math:`P_1 \\subseteq P_2` iff :math:`P_1 = \\emptyset` or
  (:math:`h_1 \\le h_2` and :math:`B_1 \\subseteq B_2`).
* :class:`OctagonRegion` -- products of 2-D octagons; tightening, emptiness, intersection
  and containment are those of :mod:`mowl.nn.el.octagon.geometry` (imported).
* :class:`UnionRegion` -- finite unions of convex pieces. Intersection distributes over
  the pieces. A union is contained in a union iff every piece is; a box is contained in a
  union of boxes iff the box minus the union is empty, decided by recursive box
  subtraction with open/closed faces (:func:`box_minus_boxes`). Pieces of other families
  are decided only when a single piece of the superset contains them.

Tolerance ``tol`` (absolute, as in :class:`~mowl.geometric.checker.ModelChecker`):
containment tests allow the subset to stick out by ``tol`` (boxes coordinatewise, balls
radially, prisms and octagons on their support values); an intersection counts as
non-empty while its regions are at most ``tol`` apart (so a disjointness axiom needs a gap
larger than ``tol``); :meth:`Region.has_interior` asks for an inscribed ball of radius
larger than ``tol / 2``.
"""

import itertools
import math

import numpy as np


class UndecidedInclusion(Exception):
    """Raised when no exact decision procedure is implemented for an inclusion."""


def _arr(x):
    return np.asarray(x, dtype=float).reshape(-1)


class Region:
    """A closed subset of :math:`\\mathbb{R}^d`. Subclasses implement :meth:`is_empty`,
    :meth:`inradius`, :meth:`_intersect`, :meth:`_contains`, :meth:`contains_points` and
    :meth:`bounding_box`."""

    #: Short name of the region family.
    family = "region"

    def __init__(self, dim):
        self.dim = int(dim)

    # -- emptiness and interior -------------------------------------------------------
    def is_empty(self, tol=0.0):
        """``True`` iff the region is empty; with ``tol > 0`` intersections whose regions
        are at most ``tol`` apart count as non-empty."""
        raise NotImplementedError

    def inradius(self):
        """Radius of the largest Euclidean ball inside the region (negative or
        ``-inf`` if the region is empty)."""
        raise NotImplementedError

    def has_interior(self, tol=0.0):
        """``True`` iff the region contains a ball of radius larger than ``tol / 2``."""
        return bool(self.inradius() > tol / 2.0)

    # -- algebra ----------------------------------------------------------------------
    def intersect(self, other, tol=0.0):
        """The intersection of ``self`` and ``other``."""
        if isinstance(other, Universe):
            return self
        if isinstance(other, EmptyRegion):
            return other
        if isinstance(other, UnionRegion):
            return other.intersect(self, tol)
        return self._intersect(other, tol)

    def _intersect(self, other, tol):
        raise UndecidedInclusion(
            f"No intersection of {type(self).__name__} and {type(other).__name__}.")

    def contains(self, other, tol=0.0):
        """``True`` iff ``other`` is a subset of ``self`` (up to ``tol``).

        :raises UndecidedInclusion: If no exact procedure is implemented for the pair.
        """
        if isinstance(other, EmptyRegion):
            return True
        if isinstance(other, Universe):
            return isinstance(self, Universe)
        if isinstance(other, UnionRegion) and not isinstance(self, UnionRegion):
            return all(self.contains(p, tol) for p in other.pieces if not p.is_empty(tol))
        if other.is_empty(tol):
            return True
        return bool(self._contains(other, tol))

    def _contains(self, other, tol):
        raise UndecidedInclusion(
            f"No inclusion test of {type(other).__name__} in {type(self).__name__}.")

    # -- sampling ---------------------------------------------------------------------
    def contains_points(self, points, tol=0.0):
        """Boolean array: which rows of ``points`` (shape ``(n, d)``) lie in the region."""
        raise NotImplementedError

    def bounding_box(self):
        """``(lo, hi)`` of an axis-parallel box containing the region."""
        raise NotImplementedError

    def sample(self, n, rng=None):
        """Up to ``n`` points of the region, by rejection from :meth:`bounding_box`."""
        rng = np.random.default_rng(rng)
        if self.is_empty():
            return np.zeros((0, self.dim))
        lo, hi = self.bounding_box()
        pts = lo + (hi - lo) * rng.random((4 * n, self.dim))
        return pts[self.contains_points(pts)][:n]


class Universe(Region):
    """The whole space (the region of :math:`\\top`)."""

    family = "universe"

    def is_empty(self, tol=0.0):
        return False

    def inradius(self):
        return math.inf

    def intersect(self, other, tol=0.0):
        return other

    def contains(self, other, tol=0.0):
        return True

    def contains_points(self, points, tol=0.0):
        return np.ones(len(points), dtype=bool)

    def bounding_box(self):
        return np.full(self.dim, -np.inf), np.full(self.dim, np.inf)


class EmptyRegion(Region):
    """The empty set (the region of :math:`\\bot`)."""

    family = "empty"

    def is_empty(self, tol=0.0):
        return True

    def inradius(self):
        return -math.inf

    def intersect(self, other, tol=0.0):
        return self

    def contains(self, other, tol=0.0):
        return other.is_empty(tol)

    def contains_points(self, points, tol=0.0):
        return np.zeros(len(points), dtype=bool)

    def bounding_box(self):
        return np.zeros(self.dim), np.zeros(self.dim)


# --------------------------------------------------------------------------------------
# Boxes
# --------------------------------------------------------------------------------------

class BoxRegion(Region):
    """Closed axis-parallel box ``[lo, hi]``; empty when ``lo[i] > hi[i]`` for some ``i``.

    Intersection, emptiness and containment follow
    :class:`mowl.geometric.checker.ModelChecker` exactly: the intersection of two boxes
    at most ``tol`` apart is the degenerate box at the larger lower corner, and
    containment allows ``tol`` per coordinate.
    """

    family = "box"

    def __init__(self, lo, hi):
        lo, hi = _arr(lo), _arr(hi)
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must have the same shape.")
        super().__init__(lo.shape[0])
        self.lo, self.hi = lo, hi

    @classmethod
    def from_center_offset(cls, center, offset):
        """Box :math:`[c - o, c + o]` (``offset`` is used as given)."""
        center, offset = _arr(center), _arr(offset)
        return cls(center - offset, center + offset)

    def as_geometric_box(self):
        """The same box as a :class:`mowl.geometric.regions.Box`."""
        from mowl.geometric.regions import Box
        return Box(self.lo, self.hi)

    def is_empty(self, tol=0.0):
        return bool(np.any(self.lo > self.hi + tol))

    def inradius(self):
        return float(np.min(self.hi - self.lo) / 2.0) if self.dim else math.inf

    def _intersect(self, other, tol):
        if not isinstance(other, BoxRegion):
            return super()._intersect(other, tol)
        if self.is_empty() or other.is_empty():
            return EmptyRegion(self.dim)
        lo = np.maximum(self.lo, other.lo)
        hi = np.minimum(self.hi, other.hi)
        if np.any(lo > hi + tol):
            return EmptyRegion(self.dim)
        return BoxRegion(lo, np.maximum(lo, hi))

    def _contains(self, other, tol):
        if isinstance(other, BoxRegion):
            return other.as_geometric_box().is_subset(self.as_geometric_box(), tol=tol)
        return super()._contains(other, tol)

    def contains_points(self, points, tol=0.0):
        points = np.atleast_2d(points)
        return np.all((points >= self.lo - tol) & (points <= self.hi + tol), axis=1)

    def bounding_box(self):
        return self.lo.copy(), self.hi.copy()

    def __repr__(self):
        return f"BoxRegion(lo={self.lo!r}, hi={self.hi!r})"


def _piece_nonempty(lo, hi, loc, hic):
    return bool(np.all((lo < hi) | ((lo == hi) & loc & hic)))


def _subtract_box(piece, blo, bhi):
    """``piece`` minus the closed box ``[blo, bhi]`` as a list of pieces.

    A piece is ``(lo, hi, lo_closed, hi_closed)``: the product of the intervals between
    ``lo`` and ``hi`` with the given closedness of each end.
    """
    lo, hi, loc, hic = (a.copy() for a in piece)
    out = []
    for i in range(len(lo)):
        if lo[i] < blo[i]:  # part with x_i < blo_i
            p = [a.copy() for a in (lo, hi, loc, hic)]
            if hi[i] >= blo[i]:
                p[1][i], p[3][i] = blo[i], False
            if _piece_nonempty(*p):
                out.append(tuple(p))
        if hi[i] > bhi[i]:  # part with x_i > bhi_i
            p = [a.copy() for a in (lo, hi, loc, hic)]
            if lo[i] <= bhi[i]:
                p[0][i], p[2][i] = bhi[i], False
            if _piece_nonempty(*p):
                out.append(tuple(p))
        if blo[i] > lo[i]:  # restrict to blo_i <= x_i <= bhi_i
            lo[i], loc[i] = blo[i], True
        if bhi[i] < hi[i]:
            hi[i], hic[i] = bhi[i], True
        if not _piece_nonempty(lo, hi, loc, hic):
            break
    return out


def box_minus_boxes(box, boxes, tol=0.0):
    """The part of the closed box ``box`` outside the union of the closed boxes ``boxes``
    (each enlarged by ``tol`` in every coordinate), as a list of pieces
    ``(lo, hi, lo_closed, hi_closed)`` (empty iff ``box`` is covered).

    Exact: pieces keep track of which faces are open, so a box covered by two boxes that
    share a face is covered.
    """
    if box.is_empty():
        return []
    d = box.dim
    pieces = [(box.lo.copy(), box.hi.copy(), np.ones(d, bool), np.ones(d, bool))]
    for b in boxes:
        if b.is_empty():
            continue
        blo, bhi = b.lo - tol, b.hi + tol
        nxt = []
        for p in pieces:
            nxt.extend(_subtract_box(p, blo, bhi))
        pieces = nxt
        if not pieces:
            break
    return pieces


# --------------------------------------------------------------------------------------
# Balls
# --------------------------------------------------------------------------------------

def _affinely_independent(points):
    if len(points) <= 1:
        return True
    diff = points[1:] - points[0]
    return np.linalg.matrix_rank(diff, tol=1e-10 * max(1.0, np.abs(diff).max())) \
        == len(points) - 1


def _ball_minmax(centers, radii, max_subsets=200000):
    """Minimiser and value of :math:`\\max_i (\\|x - c_i\\|^2 - r_i^2)`.

    Exact by support enumeration of the concave dual over the simplex (see the module
    docstring). Returns ``(x_star, value)``; the balls intersect iff ``value <= 0``.
    """
    c = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    m, d = c.shape
    if m == 1:
        return c[0].copy(), float(-r[0] ** 2)
    b = (c ** 2).sum(axis=1) - r ** 2
    max_size = min(m, d + 1)
    n_subsets = sum(math.comb(m, s) for s in range(1, max_size + 1))
    if n_subsets > max_subsets:
        raise UndecidedInclusion(
            f"Intersection of {m} balls in dimension {d}: {n_subsets} supports to enumerate.")
    best_val, best_lam, best_s = -math.inf, None, None
    for size in range(1, max_size + 1):
        for s in itertools.combinations(range(m), size):
            cs = c[list(s)]
            if not _affinely_independent(cs):
                continue
            g = cs @ cs.T
            kkt = np.zeros((size + 1, size + 1))
            kkt[:size, :size] = 2.0 * g
            kkt[:size, size] = 1.0
            kkt[size, :size] = 1.0
            rhs = np.append(b[list(s)], 1.0)
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                continue
            lam = sol[:size]
            if lam.min() < -1e-12:
                continue
            lam = np.clip(lam, 0.0, None)
            lam = lam / lam.sum()
            x = lam @ cs
            val = float(lam @ b[list(s)] - x @ x)
            if val > best_val:
                best_val, best_lam, best_s = val, lam, s
    x = best_lam @ c[list(best_s)]
    primal = float(np.max(((x - c) ** 2).sum(axis=1) - r ** 2))
    return x, primal


def lens_max_distance(c1, r1, c2, r2, e):
    """:math:`\\max \\{\\|x - e\\| : x \\in B(c_1, r_1) \\cap B(c_2, r_2)\\}` in closed form
    (``-inf`` if the lens is empty).

    A convex function attains its maximum over the compact convex lens at an extreme
    point, and every extreme point lies on one of the two spherical caps
    :math:`S_i \\cap B_j`. On :math:`S_i`, :math:`\\|c_i + r_i u - e\\|^2 = \\|a\\|^2 +
    2 r_i\\, a \\cdot u + r_i^2` with :math:`a = c_i - e`, and the cap is
    :math:`\\{u : \\|u\\| = 1,\\ u \\cdot w \\ge s\\}` with :math:`w` the unit vector towards
    :math:`c_j` and :math:`s = (r_i^2 + D^2 - r_j^2) / (2 r_i D)`. The maximum of
    :math:`a \\cdot u` over the cap is :math:`\\|a\\|` if :math:`a / \\|a\\|` is in the cap
    and otherwise attained on its rim :math:`u \\cdot w = s`:
    :math:`s\\,(a \\cdot w) + \\sqrt{1 - s^2}\\,\\|a - (a \\cdot w) w\\|`.
    """
    c1, c2, e = _arr(c1), _arr(c2), _arr(e)
    r1, r2 = float(r1), float(r2)
    v = c2 - c1
    dist = float(np.linalg.norm(v))
    if dist > r1 + r2:
        return -math.inf
    if dist + r1 <= r2:
        return float(np.linalg.norm(c1 - e)) + r1
    if dist + r2 <= r1:
        return float(np.linalg.norm(c2 - e)) + r2
    if c1.shape[0] == 1:  # intervals
        lo = max(c1[0] - r1, c2[0] - r2)
        hi = min(c1[0] + r1, c2[0] + r2)
        return max(abs(lo - e[0]), abs(hi - e[0]))
    w = v / dist
    best = -math.inf
    for ci, ri, rj, wi in ((c1, r1, r2, w), (c2, r2, r1, -w)):
        s = min(1.0, max(-1.0, (ri ** 2 + dist ** 2 - rj ** 2) / (2.0 * ri * dist)))
        a = ci - e
        na = float(np.linalg.norm(a))
        if na == 0.0:
            top = 0.0
        else:
            aw = float(a @ wi)
            if aw >= s * na:
                top = na
            else:
                perp = math.sqrt(max(na ** 2 - aw ** 2, 0.0))
                top = s * aw + math.sqrt(max(1.0 - s ** 2, 0.0)) * perp
        best = max(best, math.sqrt(max(na ** 2 + 2.0 * ri * top + ri ** 2, 0.0)))
    return best


class BallRegion(Region):
    """Closed Euclidean ball :math:`B(c, r)`, :math:`r \\ge 0` (a point when ``r = 0``).
    Never empty."""

    family = "ball"

    def __init__(self, center, radius):
        center = _arr(center)
        super().__init__(center.shape[0])
        self.center = center
        self.radius = float(radius)
        if self.radius < 0:
            raise ValueError("The radius of a ball must be non-negative.")

    def is_empty(self, tol=0.0):
        return False

    def inradius(self):
        return self.radius

    def _contains_ball(self, other, tol):
        return float(np.linalg.norm(other.center - self.center)) + other.radius \
            <= self.radius + tol

    def _intersect(self, other, tol):
        if isinstance(other, BallRegion):
            if other._contains_ball(self, 0.0):
                return self
            if self._contains_ball(other, 0.0):
                return other
            return BallIntersection([self, other])
        if isinstance(other, BallIntersection):
            return other._intersect(self, tol)
        return super()._intersect(other, tol)

    def _contains(self, other, tol):
        if isinstance(other, BallRegion):
            return self._contains_ball(other, tol)
        if isinstance(other, BallIntersection):
            if len(other.balls) != 2:
                raise UndecidedInclusion(
                    "Inclusion of an intersection of three or more balls in a ball.")
            b1, b2 = other.balls
            dist = float(np.linalg.norm(b2.center - b1.center))
            if dist > b1.radius + b2.radius:
                # Empty but at most tol apart: the closest-approach point, as for boxes.
                w = (b2.center - b1.center) / dist
                p = b1.center + w * (b1.radius + (dist - b1.radius - b2.radius) / 2.0)
                return float(np.linalg.norm(p - self.center)) <= self.radius + tol
            far = lens_max_distance(b1.center, b1.radius, b2.center, b2.radius,
                                    self.center)
            return far <= self.radius + tol
        return super()._contains(other, tol)

    def contains_points(self, points, tol=0.0):
        points = np.atleast_2d(points)
        return np.linalg.norm(points - self.center, axis=1) <= self.radius + tol

    def bounding_box(self):
        return self.center - self.radius, self.center + self.radius

    def __repr__(self):
        return f"BallRegion(center={self.center!r}, radius={self.radius!r})"


class BallIntersection(Region):
    """Intersection of two or more closed balls (not a ball in general)."""

    family = "ball"

    def __init__(self, balls):
        balls = list(balls)
        super().__init__(balls[0].dim)
        self.balls = balls

    def _arrays(self, delta=0.0):
        c = np.stack([b.center for b in self.balls])
        r = np.array([b.radius for b in self.balls]) + delta
        return c, r

    def is_empty(self, tol=0.0):
        c, r = self._arrays(tol / 2.0)
        if len(self.balls) == 2:
            return bool(np.linalg.norm(c[0] - c[1]) > r[0] + r[1])
        return bool(_ball_minmax(c, r)[1] > 0.0)

    def has_interior(self, tol=0.0):
        c, r = self._arrays(-tol / 2.0)
        if np.any(r <= 0):
            return False
        if len(self.balls) == 2:
            return bool(np.linalg.norm(c[0] - c[1]) < r[0] + r[1])
        return bool(_ball_minmax(c, r)[1] < 0.0)

    def inradius(self):
        raise UndecidedInclusion("The inradius of an intersection of balls is not "
                                 "computed; use has_interior().")

    def witness(self):
        """A point of the intersection if it is non-empty (the minimiser of the largest
        power distance)."""
        c, r = self._arrays()
        return _ball_minmax(c, r)[0]

    def _intersect(self, other, tol):
        if isinstance(other, BallRegion):
            return BallIntersection(self.balls + [other])
        if isinstance(other, BallIntersection):
            return BallIntersection(self.balls + other.balls)
        return super()._intersect(other, tol)

    def _contains(self, other, tol):
        return all(b.contains(other, tol) for b in self.balls)

    def contains_points(self, points, tol=0.0):
        out = np.ones(len(np.atleast_2d(points)), dtype=bool)
        for b in self.balls:
            out &= b.contains_points(points, tol)
        return out

    def bounding_box(self):
        lo = np.max([b.center - b.radius for b in self.balls], axis=0)
        hi = np.min([b.center + b.radius for b in self.balls], axis=0)
        return lo, hi


# --------------------------------------------------------------------------------------
# Prisms (triangle x box)
# --------------------------------------------------------------------------------------

def _prism_normals():
    from mowl.nn.el.prism.losses import NORMALS
    return NORMALS.double().numpy()


class PrismRegion(Region):
    """Prism :math:`T(h) \\times [lo, hi]` with
    :math:`T(h) = \\{p \\in \\mathbb{R}^2 : n_k \\cdot p \\le h_k\\}` for the three fixed
    unit normals of :data:`mowl.nn.el.prism.losses.NORMALS` (which sum to zero). The
    triangle is non-empty iff :math:`\\sum_k h_k \\ge 0`; it is then equilateral with
    inradius :math:`\\sum_k h_k / 3` and :math:`h` is its support function, so
    containment is :math:`h_1 \\le h_2` (Federico, Leemhuis and Kutz, DL 2026,
    Proposition 5, as implemented in :mod:`mowl.nn.el.prism.losses`)."""

    family = "prism"

    def __init__(self, h, lo, hi):
        self.h = _arr(h)
        self.lo, self.hi = _arr(lo), _arr(hi)
        super().__init__(2 + self.lo.shape[0])

    def is_empty(self, tol=0.0):
        return bool(self.h.sum() < -tol or np.any(self.lo > self.hi + tol))

    def inradius(self):
        tri = self.h.sum() / 3.0
        box = np.min(self.hi - self.lo) / 2.0 if len(self.lo) else math.inf
        return float(min(tri, box))

    def _intersect(self, other, tol):
        if isinstance(other, PrismRegion):
            return PrismRegion(np.minimum(self.h, other.h), np.maximum(self.lo, other.lo),
                               np.minimum(self.hi, other.hi))
        return super()._intersect(other, tol)

    def _contains(self, other, tol):
        if isinstance(other, PrismRegion):
            tri = np.all(other.h <= self.h + tol)
            box = np.all(self.lo - tol <= other.lo) and np.all(other.hi <= self.hi + tol)
            return bool(tri and box)
        return super()._contains(other, tol)

    def contains_points(self, points, tol=0.0):
        points = np.atleast_2d(points)
        tri = np.all(points[:, :2] @ _prism_normals().T <= self.h + tol, axis=1)
        box = np.all((points[:, 2:] >= self.lo - tol) & (points[:, 2:] <= self.hi + tol),
                     axis=1)
        return tri & box

    def triangle_vertices(self):
        """The three vertices of the triangle (meaningful if it is non-empty)."""
        n = _prism_normals()
        verts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            verts.append(np.linalg.solve(n[[i, j]], self.h[[i, j]]))
        return np.stack(verts)

    def bounding_box(self):
        v = self.triangle_vertices()
        return (np.concatenate([v.min(axis=0), self.lo]),
                np.concatenate([v.max(axis=0), self.hi]))


# --------------------------------------------------------------------------------------
# Products of octagons
# --------------------------------------------------------------------------------------

def _octagon_inradius(h):
    """Largest disc inside the 2-D octagon with bounds ``h`` (shape ``(8,)``), by LP."""
    from scipy.optimize import linprog
    from mowl.nn.el.octagon.geometry import DIRECTIONS
    n = DIRECTIONS.double().numpy()
    norms = np.linalg.norm(n, axis=1)
    a_ub = np.hstack([n, norms[:, None]])
    res = linprog(c=[0.0, 0.0, -1.0], A_ub=a_ub, b_ub=h, bounds=[(None, None)] * 3,
                  method="highs")
    if res.status == 2:  # infeasible: cannot happen (t may be negative)
        return -math.inf
    if res.status == 3:
        return math.inf
    return float(-res.fun)


class OctagonRegion(Region):
    """Product :math:`O(h^{(1)}) \\times \\dots \\times O(h^{(P)})` of 2-D octagons over the
    coordinate pairs :math:`(x_{2i}, x_{2i+1})`, bounds ``h`` of shape ``(P, 8)``, with
    the geometry of :mod:`mowl.nn.el.octagon.geometry` (Charpenay and Schockaert, IJCAI
    2024, Proposition 1)."""

    family = "octagon"

    def __init__(self, h):
        self.h = np.asarray(h, dtype=float).reshape(-1, 8)
        super().__init__(2 * self.h.shape[0])

    def _t(self):
        import torch as th
        return th.as_tensor(self.h, dtype=th.float64)

    def is_empty(self, tol=0.0):
        from mowl.nn.el.octagon import geometry as G
        return bool(G.product_emptiness_slack(self._t()) < -tol)

    def inradius(self):
        from mowl.nn.el.octagon import geometry as G
        if self.is_empty():
            return float(G.product_emptiness_slack(self._t())) / 2.0
        return min(_octagon_inradius(row) for row in self.h)

    def _intersect(self, other, tol):
        if isinstance(other, OctagonRegion):
            from mowl.nn.el.octagon import geometry as G
            return OctagonRegion(G.intersect(self._t(), other._t()).numpy())
        return super()._intersect(other, tol)

    def _contains(self, other, tol):
        if isinstance(other, OctagonRegion):
            from mowl.nn.el.octagon import geometry as G
            return bool(G.product_contains(self._t(), other._t(), tol=tol))
        return super()._contains(other, tol)

    def contains_points(self, points, tol=0.0):
        import torch as th
        from mowl.nn.el.octagon import geometry as G
        pts = th.as_tensor(np.atleast_2d(points), dtype=th.float64)
        pts = pts.reshape(pts.shape[0], -1, 2)
        return G.membership(pts, self._t(), tol=tol).all(dim=-1).numpy()

    def bounding_box(self):
        from mowl.nn.el.octagon import geometry as G
        t = G.tighten(self._t()).numpy()
        lo = np.stack([-t[:, 4], -t[:, 6]], axis=1).reshape(-1)
        hi = np.stack([t[:, 0], t[:, 2]], axis=1).reshape(-1)
        return lo, hi


# --------------------------------------------------------------------------------------
# Unions
# --------------------------------------------------------------------------------------

class UnionRegion(Region):
    """Finite union of convex pieces (all of one family)."""

    family = "union"

    def __init__(self, pieces, dim=None):
        self.pieces = [p for p in pieces if not isinstance(p, EmptyRegion)]
        if dim is None:
            if not pieces:
                raise ValueError("Give dim for an empty union.")
            dim = pieces[0].dim
        super().__init__(dim)

    def is_empty(self, tol=0.0):
        return all(p.is_empty(tol) for p in self.pieces)

    def inradius(self):
        # A finite union of closed sets without interior has no interior (Baire), so the
        # union has interior iff a piece has; the largest inscribed piece ball is a lower
        # bound of the inradius of the union and decides has_interior exactly.
        return max((p.inradius() for p in self.pieces), default=-math.inf)

    def has_interior(self, tol=0.0):
        return any(p.has_interior(tol) for p in self.pieces)

    def intersect(self, other, tol=0.0):
        if isinstance(other, Universe):
            return self
        others = other.pieces if isinstance(other, UnionRegion) else [other]
        out = []
        for p in self.pieces:
            for q in others:
                z = p.intersect(q, tol)
                if not z.is_empty(tol):
                    out.append(z)
        return UnionRegion(out, dim=self.dim)

    def contains(self, other, tol=0.0):
        if isinstance(other, Universe):
            return False
        if isinstance(other, EmptyRegion):
            return True
        subs = other.pieces if isinstance(other, UnionRegion) else [other]
        sups = [q for q in self.pieces if not q.is_empty()]
        for p in subs:
            if p.is_empty(tol):
                continue
            covered = False
            for q in sups:
                try:
                    if q.contains(p, tol):
                        covered = True
                        break
                except UndecidedInclusion:
                    pass
            if covered:
                continue
            if isinstance(p, BoxRegion) and all(isinstance(q, BoxRegion) for q in sups):
                if box_minus_boxes(p, sups, tol):
                    return False
                continue
            if not sups:
                return False
            raise UndecidedInclusion(
                f"Inclusion of a {p.family} piece in a union of {len(sups)} "
                f"{sups[0].family} pieces.")
        return True

    def contains_points(self, points, tol=0.0):
        out = np.zeros(len(np.atleast_2d(points)), dtype=bool)
        for p in self.pieces:
            out |= p.contains_points(points, tol)
        return out

    def bounding_box(self):
        boxes = [p.bounding_box() for p in self.pieces if not p.is_empty()]
        if not boxes:
            return np.zeros(self.dim), np.zeros(self.dim)
        return (np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0))

    def __repr__(self):
        return f"UnionRegion({self.pieces!r})"


# --------------------------------------------------------------------------------------
# Normal forms
# --------------------------------------------------------------------------------------

def satisfies(axiom, region, tol=0.0):
    """Decide a propositional normal form (:class:`~mowl.geometric.normal_forms.NF1` or
    :class:`~mowl.geometric.normal_forms.NF2`, with :math:`\\top` / :math:`\\bot`).

    :param region: Callable from a concept name to its :class:`Region` (use
        :class:`Universe` for :math:`\\top` and :class:`EmptyRegion` for :math:`\\bot`).
    :raises UndecidedInclusion: If the inclusion is outside the implemented procedures.
    """
    from mowl.geometric.normal_forms import NF1, NF2, BOTTOM
    if isinstance(axiom, NF1):
        left = region(axiom.sub)
    elif isinstance(axiom, NF2):
        left = region(axiom.sub1).intersect(region(axiom.sub2), tol)
    else:
        raise UndecidedInclusion(f"{type(axiom).__name__} is not a propositional normal form.")
    if axiom.sup == BOTTOM:
        return bool(left.is_empty(tol))
    return bool(region(axiom.sup).contains(left, tol))


__all__ = ["Region", "Universe", "EmptyRegion", "BoxRegion", "BallRegion",
           "BallIntersection", "PrismRegion", "OctagonRegion", "UnionRegion",
           "UndecidedInclusion", "box_minus_boxes", "lens_max_distance", "satisfies"]
