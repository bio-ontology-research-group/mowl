"""Region families of the constructed methods (:mod:`mowl.representation.construction`).

Each family is a :class:`~mowl.representation.construction.CellRegion`: a region made of
*cells* of a reading of a finite model, with the family's own point-membership test
(``contains``) and the operations it is closed under.

* :class:`ElementSets`: the cells are the domain elements themselves; a point is a vector
  over the domain and lies in a set when every element of its support does. Closed under
  intersection, union and complement (within the domain).
* :class:`AxisCones`: axis-aligned cones of :math:`\\mathbb{R}^D` (Özçep, Leemhuis & Wolter,
  JAIR 2023), given by the set of half-axes they contain; half-axis ``h`` carries cell
  ``halfaxis_cell[h]``. Intersection, polar cone (negation) and convex union
  (disjunction) act on the half-axes (JAIR Prop. 1), so on the cells.
* :class:`IntervalUnions`: unions of the disjoint intervals of a fixed grid on the line,
  one interval per cell (Lacerda, Ozaki & Guimarães, TGDK 2024, Def. 12). Closed under
  intersection, union and complement (relative to the union of all intervals).
* :class:`SimplexFaces`: faces of the standard simplex (convex hulls of vertices, one
  vertex per cell; Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 3), optionally thickened
  to full dimension (``eps``). Closed under intersection only.
"""

import numpy as np
import torch as th

from mowl.representation.construction import CellRegion
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["ElementSets", "AxisCones", "IntervalUnions", "SimplexFaces"]


def _first(mask):
    """``(B, n)`` at time 0."""
    return mask[:, 0]


class ElementSets(CellRegion):
    """Sets of domain elements. A point is a vector ``(P, n)`` over the domain; it lies in a
    set when it is not zero and every element of its support belongs to the set (a point
    of one element is its indicator vector)."""

    union_closed = True
    complement_closed = True

    def contains(self, points, time=0):
        x = points.to(th.float64).reshape(points.shape[0], -1)
        support = x != 0                                              # (P, n)
        mask = self.at(time)[:, 0]                                    # (B, n)
        outside = (support.unsqueeze(0) & ~mask.unsqueeze(1)).any(-1)
        return ~outside & support.any(-1).unsqueeze(0)

    def point(self, cells):
        out = th.zeros(self.n_cells, dtype=th.float64)
        out[sorted(cells)] = 1.0
        return out


class AxisCones(CellRegion):
    """Axis-aligned cones (al-cones).

    :param halfaxis_cell: ``(2 D,)`` long: the cell of every half-axis (half-axis ``2 i``
        is :math:`\\mathbb{R}^+ e_i`, ``2 i + 1`` is :math:`\\mathbb{R}^- e_i`).
    :param doubled: Every coordinate of the cells' space is doubled (JAIR Prop. 8), so
        that a point can lie on any set of half-axes of the original space (an individual
        known to be one of several elements).

    The bottom cone :math:`\\{0\\}` plays the empty set (:meth:`is_empty`): it holds no
    ray, and individuals are rays (non-zero points).
    """

    union_closed = True
    complement_closed = True

    @property
    def halfaxis_cell(self):
        return self.geometry["halfaxis_cell"]

    @property
    def dim(self):
        return self.halfaxis_cell.shape[0] // 2

    @property
    def bits(self):
        """``(B, T, 2 D)`` boolean: the half-axes of the cone at every time point."""
        return self.mask[..., self.halfaxis_cell]

    def contains(self, points, time=0, tol=1e-12):
        x = points.to(th.float64).reshape(points.shape[0], -1)
        if x.shape[1] != self.dim:
            raise ValueError(f"Points of R^{x.shape[1]} in cones of R^{self.dim}.")
        support = th.zeros(x.shape[0], 2 * self.dim, dtype=th.bool)
        support[:, 0::2] = x > tol
        support[:, 1::2] = x < -tol
        bits = self.at(time)[:, 0][..., self.halfaxis_cell]          # (B, 2D)
        return ~(support.unsqueeze(0) & ~bits.unsqueeze(1)).any(-1)

    def point(self, cells):
        """The ray of the first half-axis of the cell (one cell); for several cells in a
        doubled space, the point whose support is the half-axes of all of them (JAIR
        Prop. 8)."""
        cells = sorted(cells)
        if not self.geometry.get("doubled", False):
            if len(cells) != 1:
                raise ConstructorUnavailable(
                    "A point on one half-axis stands for one cell; an individual of several "
                    "cells needs the doubled space (doubled=True).")
            h = int(th.nonzero(self.halfaxis_cell == cells[0])[0])
            v = th.zeros(self.dim, dtype=th.float64)
            v[h // 2] = 1.0 if h % 2 == 0 else -1.0
            return v
        original = self.geometry["original_cell"]                     # (2 D / 2,)
        support = {h for h in range(original.shape[0]) if int(original[h]) in cells}
        half = original.shape[0] // 2
        v = th.zeros(2 * half, dtype=th.float64)
        for i in range(half):
            p, m = (2 * i) in support, (2 * i + 1) in support
            if p and m:
                v[2 * i], v[2 * i + 1] = 1.0, -1.0
            elif p:
                v[2 * i] = v[2 * i + 1] = 1.0
            elif m:
                v[2 * i] = v[2 * i + 1] = -1.0
        return v

    def tensors(self):
        return {"bits": self.bits}


class IntervalUnions(CellRegion):
    """Unions of intervals of a fixed grid of disjoint intervals of the line, one per cell.

    :param lower, upper: ``(n,)`` float64: the grid (pairwise disjoint, non-empty).
    """

    union_closed = True
    complement_closed = True

    @property
    def lower(self):
        return self.geometry["lower"]

    @property
    def upper(self):
        return self.geometry["upper"]

    def contains(self, points, time=0):
        x = points.to(th.float64).reshape(-1, 1)                      # (P, 1)
        cell = (x >= self.lower) & (x <= self.upper)                   # (P, n)
        mask = self.at(time)[:, 0]                                    # (B, n)
        return (cell.unsqueeze(0) & mask.unsqueeze(1)).any(-1)

    def point(self, cells):
        if len(cells) != 1:
            raise ConstructorUnavailable("A point of the line lies in one interval; an "
                                         "individual of several cells has no point.")
        (k,) = cells
        return ((self.lower[k] + self.upper[k]) / 2).reshape(1)

    def intervals(self, row=0, time=0):
        """The intervals ``[(lower, upper), ...]`` of one row, in order."""
        ids = th.nonzero(self.mask[row, time]).reshape(-1).tolist()
        return [(float(self.lower[k]), float(self.upper[k])) for k in ids]

    def tensors(self):
        active = self.mask[:, 0]
        nan = th.full(active.shape, float("nan"), dtype=th.float64)
        return {"lower": th.where(active, self.lower, nan),
                "upper": th.where(active, self.upper, nan)}


class SimplexFaces(CellRegion):
    """Faces of the standard simplex: the convex hull of the vertices of the cells.

    :param points: ``(n, n - 1)`` float64: the vertices
        (:func:`~mowl.geometric.regions.simplex_points`).
    :param eps: ``0``: faces (lower-dimensional for a proper subset of the cells);
        ``eps > 0``: the thickened faces :math:`\\{x : \\lambda(x) \\ge 0,\\ \\lambda_j(x)
        \\le \\varepsilon \\text{ for } j \\notin S\\}` of
        :class:`~mowl.geometric.regions.ThickFace`, with :math:`\\lambda` the barycentric
        coordinates (full-dimensional for a non-empty set of cells).

    Inclusion and intersection are decided on the cells (exact: ``conv(S) ⊆ conv(T)`` iff
    ``S ⊆ T``, ``conv(S) ∩ conv(T) = conv(S ∩ T)``, and the same for thickened faces of
    one ``eps < 1 / n``). The union of two faces is not convex; ``exists r.C`` is a face
    only for faces (``eps = 0``).
    """

    @property
    def exists_exact(self):
        return float(self.geometry.get("eps", 0.0)) == 0.0

    @property
    def points(self):
        return self.geometry["points"]

    @property
    def eps(self):
        return float(self.geometry.get("eps", 0.0))

    def _inverse(self):
        inv = self.geometry.get("_inverse")
        if inv is None:
            n = self.points.shape[0]
            a = th.cat([self.points.T, th.ones(1, n, dtype=th.float64)], 0)   # (n, n)
            inv = th.linalg.inv(a)
            self.geometry["_inverse"] = inv
        return inv

    def barycentric(self, points):
        """``(P, n)``: the barycentric coordinates of the points."""
        x = points.to(th.float64).reshape(points.shape[0], -1)
        rhs = th.cat([x, th.ones(x.shape[0], 1, dtype=th.float64)], 1)        # (P, n)
        return rhs @ self._inverse().T

    def is_empty(self, time=None):
        empty = super().is_empty(time)
        if self.eps * self.n_cells >= 1:        # all lambda <= eps can still sum to one
            return th.zeros_like(empty)
        return empty

    def contains(self, points, time=0, tol=1e-9):
        lam = self.barycentric(points)                                 # (P, n)
        mask = self.at(time)[:, 0]                                     # (B, n)
        ok = (lam >= -tol).all(-1)                                     # (P,)
        out_of = lam.unsqueeze(0) > self.eps + tol                     # (1, P, n)
        bad = (out_of & ~mask.unsqueeze(1)).any(-1)                    # (B, P)
        return ok.unsqueeze(0) & ~bad

    def point(self, cells):
        """The centroid of the vertices of the cells."""
        ids = sorted(cells)
        return self.points[ids].mean(0)

    def vertex_ids(self, row=0, time=0):
        return frozenset(th.nonzero(self.mask[row, time]).reshape(-1).tolist())

    def tensors(self):
        return {"vertices": self.mask[:, 0].to(th.float64), "points": self.points}


def simplex(n):
    """The vertices of the standard simplex with ``n`` vertices, ``(n, n - 1)``
    float64."""
    from mowl.geometric.regions import simplex_points
    return th.as_tensor(np.asarray(simplex_points(n), dtype=np.float64))
