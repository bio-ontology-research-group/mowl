"""Pure-numpy geometric region types for :mod:`mowl.geometric`.

Semantics and guarantees:
- :class:`IntervalUnion`: finite union of closed 1-D intervals
  :math:`\\bigcup_i [a_i, b_i] \\subseteq \\mathbb{R}`. The constructor drops
  empty intervals (:math:`a_i > b_i`), sorts, and merges overlapping or touching
  intervals so ``intervals`` is always canonical (closed, pairwise disjoint,
  strictly separated, sorted). Equality compares normalised interval lists with
  an endpoint tolerance of ``1e-12``.
- :class:`RectUnion`: finite union of closed 2-D axis-aligned rectangles
  :math:`\\bigcup_k (I_k \\times J_k) \\subseteq \\mathbb{R}^2` interpreting
  binary roles in 1-D interval models (Spec §5.2, §5.3). Provides exact
  domain/range projection (:meth:`RectUnion.exists_left`,
  :meth:`RectUnion.exists_right`), relational composition
  (:meth:`RectUnion.compose`) for role chains (:class:`~mowl.geometric.normal_forms.NF7`),
  and exact 2-D inclusion (:meth:`RectUnion.is_subset`) for role inclusions
  (:class:`~mowl.geometric.normal_forms.NF6`).
- :class:`Box`: closed axis-aligned hyperrectangle
  :math:`[\\text{lo}, \\text{hi}] \\subseteq \\mathbb{R}^d`. A box is empty iff
  ``lo[i] > hi[i]`` on at least one coordinate; :meth:`Box.empty` constructs the
  canonical empty box with ``lo = +inf`` and ``hi = -inf``, and an empty box is
  a subset of every box of the same dimension.
- :func:`simplex_points`: returns the :math:`n` vertices of the standard
  simplex in :math:`\\mathbb{R}^{n-1}` as an :math:`(n, n-1)` float array whose
  row :math:`i` (:math:`0 \\le i < n-1`) is the unit vector :math:`e_i` and
  whose last row :math:`n-1` is the zero vector (shape ``(1, 0)`` when
  :math:`n = 1`).
- :class:`Face`: face of the standard simplex in :math:`\\mathbb{R}^k`
  (:math:`k = n - 1`) spanned by a subset of vertex indices
  ``vertex_ids``. Set inclusion (:meth:`Face.is_subset`) and intersection
  (:meth:`Face.intersection`) are computed combinatorially on ``vertex_ids``;
  this is exact for faces of one simplex, which is the only use in the
  polyhedral construction (Gutiérrez-Basulto & Schockaert 2018, Prop. 3).
  :meth:`Face.h_rep` returns ``(A_ub, b_ub, A_eq, b_eq)`` in the ambient
  simplex coordinates using barycentric coordinates
  :math:`\\lambda_j(x) = x_j` for :math:`j < n-1` and
  :math:`\\lambda_{n-1}(x) = 1 - \\sum_{j < n-1} x_j` (encoded via row
  :math:`(-1, \\dots, -1)` with constant :math:`1`): equality rows
  :math:`\\lambda_j(x) = 0` for every vertex :math:`j \\notin \\text{vertex\\_ids}`
  and inequality rows :math:`-\\lambda_j(x) \\le 0` for every vertex
  :math:`j \\in \\text{vertex\\_ids}`.
- :class:`PairHull`: V-polytope of concatenated vertex pairs
  :math:`v_i \\oplus v_j \\in \\mathbb{R}^{2k}` stored as an :math:`(m, 2k)`
  array ``points``, with :meth:`PairHull.left` and :meth:`PairHull.right`
  returning the :math:`(m, k)` left and right halves for the model checker's
  linear programs.
"""
from __future__ import annotations

from typing import FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

_INTERVAL_EQ_TOL: float = 1e-12


class IntervalUnion:
    """Finite union of closed 1-D intervals :math:`\\bigcup_i [a_i, b_i]`.

    All intervals are closed. The constructor drops empty intervals
    (``lo > hi``), sorts by lower then upper endpoint, and merges overlapping
    or touching intervals (``next_lo <= cur_hi``) so ``self.intervals`` is
    always sorted and pairwise disjoint. Equality compares the normalised
    interval lists with a tolerance of ``1e-12``.
    """

    def __init__(
        self, intervals: Iterable[Tuple[float, float]] = ()
    ) -> None:
        valid: List[Tuple[float, float]] = []
        for lo, hi in intervals:
            f_lo = float(lo)
            f_hi = float(hi)
            if f_lo <= f_hi:
                valid.append((f_lo, f_hi))

        valid.sort(key=lambda ivl: (ivl[0], ivl[1]))

        merged: List[Tuple[float, float]] = []
        for lo, hi in valid:
            if not merged:
                merged.append((lo, hi))
            else:
                prev_lo, prev_hi = merged[-1]
                if lo <= prev_hi:
                    merged[-1] = (prev_lo, max(prev_hi, hi))
                else:
                    merged.append((lo, hi))

        self.intervals: List[Tuple[float, float]] = merged

    @property
    def is_empty(self) -> bool:
        """Return ``True`` iff this union contains no points."""
        return len(self.intervals) == 0

    def contains(self, x: float) -> bool:
        """Return ``True`` iff ``x`` lies in at least one closed interval."""
        val = float(x)
        for lo, hi in self.intervals:
            if lo <= val <= hi:
                return True
            if lo > val:
                break
        return False

    def intersection(self, other: IntervalUnion) -> IntervalUnion:
        """Return the exact intersection of two closed :class:`IntervalUnion` objects."""
        out: List[Tuple[float, float]] = []
        i = 0
        j = 0
        a = self.intervals
        b = other.intervals
        while i < len(a) and j < len(b):
            lo = max(a[i][0], b[j][0])
            hi = min(a[i][1], b[j][1])
            if lo <= hi:
                out.append((lo, hi))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return IntervalUnion(out)

    def union(self, other: IntervalUnion) -> IntervalUnion:
        """Return the union of ``self`` and ``other``."""
        return IntervalUnion(self.intervals + other.intervals)

    def is_subset(self, other: IntervalUnion) -> bool:
        """Return ``True`` iff ``self`` is a subset of ``other``."""
        if self.is_empty:
            return True
        if other.is_empty:
            return False
        for a_lo, a_hi in self.intervals:
            covered = False
            for b_lo, b_hi in other.intervals:
                if b_lo - _INTERVAL_EQ_TOL <= a_lo and a_hi <= b_hi + _INTERVAL_EQ_TOL:
                    covered = True
                    break
            if not covered:
                return False
        return True

    def intersects(self, other: IntervalUnion) -> bool:
        """Return ``True`` iff ``self`` and ``other`` have non-empty intersection."""
        return not self.intersection(other).is_empty

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalUnion):
            return NotImplemented
        if len(self.intervals) != len(other.intervals):
            return False
        for (l1, h1), (l2, h2) in zip(self.intervals, other.intervals):
            if abs(l1 - l2) > _INTERVAL_EQ_TOL or abs(h1 - h2) > _INTERVAL_EQ_TOL:
                return False
        return True

    def __repr__(self) -> str:
        return f"IntervalUnion({self.intervals!r})"


EMPTY_1D: IntervalUnion = IntervalUnion([])


class RectUnion:
    """Finite union of closed 2-D rectangles :math:`\\bigcup_k (I_k \\times J_k)`.

    Each rectangle is a pair ``((i_lo, i_hi), (j_lo, j_hi))`` of closed 1-D
    intervals. Empty rectangles (where ``i_lo > i_hi`` or ``j_lo > j_hi``) are
    dropped on construction.
    """

    def __init__(
        self,
        rects: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]] = (),
    ) -> None:
        cleaned: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        seen: set = set()
        for (i_lo, i_hi), (j_lo, j_hi) in rects:
            fi0, fi1 = float(i_lo), float(i_hi)
            fj0, fj1 = float(j_lo), float(j_hi)
            if fi0 <= fi1 and fj0 <= fj1:
                rect = ((fi0, fi1), (fj0, fj1))
                if rect not in seen:
                    seen.add(rect)
                    cleaned.append(rect)
        self.rects: List[Tuple[Tuple[float, float], Tuple[float, float]]] = cleaned

    @property
    def is_empty(self) -> bool:
        """Return ``True`` iff this rectangle union contains no rectangles."""
        return len(self.rects) == 0

    def exists_left(self, cond: IntervalUnion) -> IntervalUnion:
        """Return :math:`\\bigcup \\{I : (I \\times J) \\in \\text{self}, J \\cap \\text{cond} \\neq \\emptyset\\}`."""
        if cond.is_empty or self.is_empty:
            return EMPTY_1D
        matched: List[Tuple[float, float]] = []
        for i_ivl, j_ivl in self.rects:
            if IntervalUnion([j_ivl]).intersects(cond):
                matched.append(i_ivl)
        return IntervalUnion(matched)

    def exists_right(self, cond: IntervalUnion) -> IntervalUnion:
        """Return :math:`\\bigcup \\{J : (I \\times J) \\in \\text{self}, I \\cap \\text{cond} \\neq \\emptyset\\}`."""
        if cond.is_empty or self.is_empty:
            return EMPTY_1D
        matched: List[Tuple[float, float]] = []
        for i_ivl, j_ivl in self.rects:
            if IntervalUnion([i_ivl]).intersects(cond):
                matched.append(j_ivl)
        return IntervalUnion(matched)

    def compose(self, other: RectUnion) -> RectUnion:
        """Return the relational composition ``self o other``.

        Collects all rectangle pairs ``(I, K)`` for which some ``I x J`` in
        ``self`` and ``J' x K`` in ``other`` satisfy :math:`J \\cap J' \\neq \\emptyset`.
        """
        if self.is_empty or other.is_empty:
            return RectUnion([])
        composed: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for i_ivl, j_ivl in self.rects:
            j_lo, j_hi = j_ivl
            for jp_ivl, k_ivl in other.rects:
                if max(j_lo, jp_ivl[0]) <= min(j_hi, jp_ivl[1]):
                    composed.append((i_ivl, k_ivl))
        return RectUnion(composed)

    def union(self, other: RectUnion) -> RectUnion:
        """Return the union of two :class:`RectUnion` objects."""
        return RectUnion(self.rects + other.rects)

    def is_subset(self, other: RectUnion) -> bool:
        """Return ``True`` iff every rectangle in ``self`` is covered by ``other``."""
        if self.is_empty:
            return True
        if other.is_empty:
            return False

        for (i_lo, i_hi), j_ivl in self.rects:
            j_target = IntervalUnion([j_ivl])
            # Collect critical x-coordinates splitting [i_lo, i_hi]
            xs = {i_lo, i_hi}
            for (oi_lo, oi_hi), _ in other.rects:
                if i_lo < oi_lo < i_hi:
                    xs.add(oi_lo)
                if i_lo < oi_hi < i_hi:
                    xs.add(oi_hi)
            sorted_x = sorted(xs)
            slabs: List[Tuple[float, float]] = (
                [(i_lo, i_hi)]
                if len(sorted_x) == 1
                else [
                    (sorted_x[idx], sorted_x[idx + 1])
                    for idx in range(len(sorted_x) - 1)
                ]
            )
            for s_lo, s_hi in slabs:
                covering_js = [
                    oj_ivl
                    for (oi_lo, oi_hi), oj_ivl in other.rects
                    if oi_lo - _INTERVAL_EQ_TOL <= s_lo
                    and s_hi <= oi_hi + _INTERVAL_EQ_TOL
                ]
                if not j_target.is_subset(IntervalUnion(covering_js)):
                    return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RectUnion):
            return NotImplemented
        return self.is_subset(other) and other.is_subset(self)

    def __repr__(self) -> str:
        return f"RectUnion({self.rects!r})"


class Box:
    """Closed axis-aligned box :math:`[\\text{lo}, \\text{hi}] \\subseteq \\mathbb{R}^d`.

    A box is empty when ``lo[i] > hi[i]`` for at least one coordinate ``i``.
    :meth:`Box.empty` constructs the canonical empty box with ``lo = +inf`` and
    ``hi = -inf``. Inclusion (:meth:`is_subset`) of an empty box in any box is
    ``True``, and membership (:meth:`contains`) is closed.
    """

    def __init__(self, lo: Sequence[float], hi: Sequence[float]) -> None:
        self.lo: np.ndarray = np.asarray(lo, dtype=float).reshape(-1)
        self.hi: np.ndarray = np.asarray(hi, dtype=float).reshape(-1)
        if self.lo.shape != self.hi.shape:
            raise ValueError(
                f"Box lo and hi shapes must match, got {self.lo.shape} vs {self.hi.shape}"
            )

    @classmethod
    def empty(cls, d: int) -> Box:
        """Return the canonical empty :math:`d`-dimensional box (``lo = +inf, hi = -inf``)."""
        return cls(
            np.full(d, np.inf, dtype=float),
            np.full(d, -np.inf, dtype=float),
        )

    @property
    def is_empty(self) -> bool:
        """Return ``True`` iff ``lo[i] > hi[i]`` for any coordinate ``i``."""
        return bool(np.any(self.lo > self.hi))

    def contains(self, point: Sequence[float], tol: float = 0.0) -> bool:
        """Return ``True`` iff ``point`` lies in the closed box ``[lo, hi]``."""
        if self.is_empty:
            return False
        pt = np.asarray(point, dtype=float).reshape(-1)
        return bool(
            np.all(pt >= self.lo - tol) and np.all(pt <= self.hi + tol)
        )

    def intersection(self, other: Box) -> Box:
        """Return the intersection of two closed :class:`Box` objects."""
        d = int(self.lo.shape[0])
        if self.is_empty or other.is_empty:
            return Box.empty(d)
        lo = np.maximum(self.lo, other.lo)
        hi = np.minimum(self.hi, other.hi)
        if np.any(lo > hi):
            return Box.empty(d)
        return Box(lo, hi)

    def is_subset(self, other: Box, tol: float = 1e-12) -> bool:
        """Return ``True`` iff ``self`` is a subset of ``other`` (empty box is a subset of any box)."""
        if self.is_empty:
            return True
        if other.is_empty:
            return False
        return bool(
            np.all(other.lo - tol <= self.lo)
            and np.all(self.hi <= other.hi + tol)
        )

    def intersects(self, other: Box) -> bool:
        """Return ``True`` iff ``self`` and ``other`` have non-empty intersection."""
        return not self.intersection(other).is_empty

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        if self.is_empty and other.is_empty:
            return self.lo.shape == other.lo.shape
        if self.is_empty != other.is_empty:
            return False
        return bool(
            np.allclose(self.lo, other.lo, atol=1e-12, rtol=0.0)
            and np.allclose(self.hi, other.hi, atol=1e-12, rtol=0.0)
        )

    def __repr__(self) -> str:
        return f"Box(lo={self.lo!r}, hi={self.hi!r})"


def simplex_points(n: int) -> np.ndarray:
    """Return the :math:`n` vertices of the standard simplex in :math:`\\mathbb{R}^{n-1}`.

    Row ``i`` (0-based, ``i < n - 1``) is the unit vector :math:`e_i`; row
    ``n - 1`` is the zero vector. For ``n = 1``, returns an array of shape
    ``(1, 0)``.
    """
    if n < 1:
        raise ValueError(f"simplex_points requires n >= 1, got {n}")
    k = n - 1
    pts = np.zeros((n, k), dtype=float)
    if k > 0:
        pts[:k, :k] = np.eye(k, dtype=float)
    return pts


def _canonical_simplex_vertex(dim: int, vid: int) -> np.ndarray:
    """Return vertex ``vid`` of ``simplex_points(dim + 1)``."""
    v = np.zeros(dim, dtype=float)
    if 0 <= vid < dim:
        v[vid] = 1.0
    elif vid != dim:
        raise ValueError(f"Vertex id {vid} out of range for ambient dimension {dim} (n={dim + 1})")
    return v


class Face:
    """Face of the standard simplex in :math:`\\mathbb{R}^{\\text{dim}}` (:math:`n = \\text{dim} + 1`).

    Semantics:
    - Represents :math:`F(S) = \\text{conv}\\{v_j : j \\in S\\}` where
      :math:`S = \\text{vertex\\_ids} \\subseteq \\{0, \\dots, n - 1\\}` and
      :math:`v_0, \\dots, v_{n-1}` are the rows of :func:`simplex_points`.
    - :meth:`is_subset` and :meth:`intersection` are computed as exact set
      operations on ``vertex_ids`` (:math:`F(S) \\subseteq F(S') \\iff S \\subseteq S'`
      and :math:`F(S) \\cap F(S') = F(S \\cap S')`); this is exact for faces of
      a single simplex, which is the only use in :mod:`mowl.geometric`.
    - :meth:`h_rep` returns ``(A_ub, b_ub, A_eq, b_eq)`` in the ambient
      coordinates :math:`x \\in \\mathbb{R}^{\\text{dim}}`. With barycentric
      coordinates :math:`\\lambda_j(x) = x_j` (:math:`j < n - 1`) and
      :math:`\\lambda_{n-1}(x) = (-1, \\dots, -1) \\cdot x + 1 = 1 - \\sum_{j < n-1} x_j`,
      every vertex :math:`j \\notin S` yields an equality row
      :math:`\\lambda_j(x) = 0` in ``(A_eq, b_eq)``, and every vertex
      :math:`j \\in S` yields an inequality row :math:`-\\lambda_j(x) \\le 0`
      in ``(A_ub, b_ub)``.
    """

    def __init__(
        self,
        dim: Union[int, np.ndarray],
        vertices: Optional[Union[np.ndarray, Iterable[int]]] = None,
        vertex_ids: Optional[Iterable[int]] = None,
    ) -> None:
        # Support both Face(dim, vertices, vertex_ids) and Face(simplex_pts, vertex_ids)
        if isinstance(dim, np.ndarray):
            simplex_arr = np.asarray(dim, dtype=float)
            if simplex_arr.ndim != 2:
                raise ValueError(f"Expected 2-D simplex array, got shape {simplex_arr.shape}")
            self.dim: int = int(simplex_arr.shape[1])
            vids_iter: Iterable[int] = (
                vertices if vertex_ids is None else vertex_ids  # type: ignore[assignment]
            )
            sorted_ids = sorted(int(i) for i in (vids_iter or ()))
            self.vertex_ids: FrozenSet[int] = frozenset(sorted_ids)
            if sorted_ids:
                self.vertices: np.ndarray = np.asarray(
                    simplex_arr[sorted_ids], dtype=float
                ).reshape((len(sorted_ids), self.dim))
            else:
                self.vertices = np.zeros((0, self.dim), dtype=float)
            return

        self.dim = int(dim)
        if vertex_ids is None and vertices is not None and not isinstance(vertices, np.ndarray):
            # Called as Face(dim, vertex_ids_set)
            sorted_ids = sorted(int(i) for i in vertices)
            self.vertex_ids = frozenset(sorted_ids)
            if sorted_ids:
                self.vertices = np.vstack(
                    [_canonical_simplex_vertex(self.dim, i) for i in sorted_ids]
                )
            else:
                self.vertices = np.zeros((0, self.dim), dtype=float)
            return

        self.vertex_ids = frozenset(int(i) for i in (vertex_ids or ()))
        if vertices is None:
            sorted_ids = sorted(self.vertex_ids)
            if sorted_ids:
                self.vertices = np.vstack(
                    [_canonical_simplex_vertex(self.dim, i) for i in sorted_ids]
                )
            else:
                self.vertices = np.zeros((0, self.dim), dtype=float)
        else:
            arr = np.asarray(vertices, dtype=float)
            if arr.size == 0:
                self.vertices = np.zeros((0, self.dim), dtype=float)
            else:
                self.vertices = arr.reshape((-1, self.dim))

    @classmethod
    def from_simplex(
        cls, simplex: np.ndarray, vertex_ids: Iterable[int]
    ) -> Face:
        """Construct a :class:`Face` from ``simplex_points(n)`` and ``vertex_ids``."""
        simplex_arr = np.asarray(simplex, dtype=float)
        k = int(simplex_arr.shape[1])
        sorted_ids = sorted(int(i) for i in vertex_ids)
        vids = frozenset(sorted_ids)
        if sorted_ids:
            verts = simplex_arr[sorted_ids].reshape((len(sorted_ids), k))
        else:
            verts = np.zeros((0, k), dtype=float)
        return cls(k, verts, vids)

    @property
    def is_empty(self) -> bool:
        """Return ``True`` iff this face has no vertices."""
        return len(self.vertex_ids) == 0

    def intersection(self, other: Face) -> Face:
        """Return the intersection of two faces of the same simplex (exact via ``vertex_ids``)."""
        if self.dim != other.dim:
            raise ValueError(
                f"Cannot intersect faces of different ambient dimensions: {self.dim} vs {other.dim}"
            )
        common_ids = self.vertex_ids & other.vertex_ids
        sorted_ids = sorted(common_ids)
        if not sorted_ids:
            return Face(
                self.dim,
                np.zeros((0, self.dim), dtype=float),
                frozenset(),
            )
        verts = np.vstack(
            [_canonical_simplex_vertex(self.dim, i) for i in sorted_ids]
        )
        return Face(self.dim, verts, common_ids)

    def is_subset(self, other: Face) -> bool:
        """Return ``True`` iff ``self.vertex_ids <= other.vertex_ids`` (exact for faces of one simplex)."""
        if self.dim != other.dim:
            raise ValueError(
                f"Cannot compare faces of different ambient dimensions: {self.dim} vs {other.dim}"
            )
        return self.vertex_ids.issubset(other.vertex_ids)

    def h_rep(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(A_ub, b_ub, A_eq, b_eq)`` of the face in ambient simplex coordinates.

        For ambient dimension ``k = self.dim`` and ``n = k + 1`` simplex vertices:
        - Barycentric coordinate :math:`\\lambda_j(x) = a_j \\cdot x + c_j` has
          :math:`a_j = e_j, c_j = 0` for :math:`0 \\le j < n - 1` and
          :math:`a_{n-1} = (-1, \\dots, -1), c_{n-1} = 1` for :math:`j = n - 1`.
        - For every :math:`j \\notin \\text{vertex\\_ids}`, equality
          :math:`\\lambda_j(x) = 0` is encoded as row :math:`a_j` with RHS
          :math:`-c_j` in ``(A_eq, b_eq)`` (:math:`a_j \\cdot x + c_j = 0`).
        - For every :math:`j \\in \\text{vertex\\_ids}`, inequality
          :math:`-\\lambda_j(x) \\le 0` is encoded as row :math:`-a_j` with RHS
          :math:`c_j` in ``(A_ub, b_ub)`` (:math:`-a_j \\cdot x \\le c_j`).
        """
        k = self.dim
        n = k + 1
        ub_rows: List[np.ndarray] = []
        ub_rhs: List[float] = []
        eq_rows: List[np.ndarray] = []
        eq_rhs: List[float] = []

        for j in range(n):
            if j < k:
                a_j = np.zeros(k, dtype=float)
                a_j[j] = 1.0
                c_j = 0.0
            else:
                a_j = np.full(k, -1.0, dtype=float)
                c_j = 1.0

            if j in self.vertex_ids:
                # -lambda_j(x) <= 0  <=>  -a_j . x <= c_j
                ub_rows.append(-a_j)
                ub_rhs.append(c_j)
            else:
                # lambda_j(x) = 0  <=>  a_j . x = -c_j
                eq_rows.append(a_j)
                eq_rhs.append(-c_j)

        A_ub = (
            np.vstack(ub_rows)
            if ub_rows
            else np.zeros((0, k), dtype=float)
        )
        b_ub = np.asarray(ub_rhs, dtype=float)
        A_eq = (
            np.vstack(eq_rows)
            if eq_rows
            else np.zeros((0, k), dtype=float)
        )
        b_eq = np.asarray(eq_rhs, dtype=float)
        return A_ub, b_ub, A_eq, b_eq

    def contains(self, point: Sequence[float], tol: float = 1e-9) -> bool:
        """Return ``True`` iff ``point`` satisfies :meth:`h_rep` within ``tol``."""
        if self.is_empty:
            return False
        pt = np.asarray(point, dtype=float).reshape(self.dim)
        A_ub, b_ub, A_eq, b_eq = self.h_rep()
        if A_ub.shape[0] > 0 and bool(np.any(A_ub @ pt - b_ub > tol)):
            return False
        if A_eq.shape[0] > 0 and bool(np.any(np.abs(A_eq @ pt - b_eq) > tol)):
            return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Face):
            return NotImplemented
        return self.dim == other.dim and self.vertex_ids == other.vertex_ids

    def __repr__(self) -> str:
        return f"Face(dim={self.dim}, vertex_ids={sorted(self.vertex_ids)!r})"


class PairHull:
    """Polyhedral role region storing generating pairs :math:`v_i \\oplus v_j \\in \\mathbb{R}^{2k}`.

    Attributes
    ----------
    points : np.ndarray
        Array of shape ``(m, 2k)`` whose rows are concatenated vertex pairs
        :math:`v_i \\oplus v_j`. May have ``m = 0`` for an empty role region.
    """

    def __init__(self, points: np.ndarray) -> None:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] % 2 != 0:
            raise ValueError(
                f"PairHull.points must have 2-D shape (m, 2k), got {arr.shape}"
            )
        self.points: np.ndarray = arr

    @property
    def is_empty(self) -> bool:
        """Return ``True`` iff ``self.points`` has zero rows."""
        return int(self.points.shape[0]) == 0

    def left(self) -> np.ndarray:
        """Return the ``(m, k)`` array of first-factor points :math:`v_i`."""
        k = self.points.shape[1] // 2
        return self.points[:, :k]

    def right(self) -> np.ndarray:
        """Return the ``(m, k)`` array of second-factor points :math:`v_j`."""
        k = self.points.shape[1] // 2
        return self.points[:, k:]

    def __repr__(self) -> str:
        return f"PairHull(shape={self.points.shape!r})"
