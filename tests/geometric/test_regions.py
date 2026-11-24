"""Tests for pure-geometry region types in mowl.geometric.regions."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from mowl.geometric.regions import (
    EMPTY_1D,
    Box,
    Face,
    IntervalUnion,
    PairHull,
    RectUnion,
    simplex_points,
)


def test_interval_union_algebra() -> None:
    # Merging of touching and overlapping intervals; dropping empty (lo > hi)
    u_touch = IntervalUnion([(0.0, 1.0), (1.0, 2.0)])
    assert u_touch == IntervalUnion([(0.0, 2.0)])
    assert u_touch.intervals == [(0.0, 2.0)]

    u_overlap_unsorted = IntervalUnion(
        [(3.0, 2.0), (2.0, 4.0), (0.0, 2.5), (5.0, 5.0)]
    )
    assert u_overlap_unsorted.intervals == [(0.0, 4.0), (5.0, 5.0)]
    assert not u_overlap_unsorted.is_empty
    assert u_overlap_unsorted.contains(5.0)
    assert u_overlap_unsorted.contains(0.0)
    assert u_overlap_unsorted.contains(4.0)
    assert not u_overlap_unsorted.contains(4.5)
    assert not u_overlap_unsorted.contains(-0.1)

    # Empty constant
    assert EMPTY_1D.is_empty
    assert EMPTY_1D.intervals == []
    assert not EMPTY_1D.contains(0.0)
    assert IntervalUnion([(2.0, 1.0)]) == EMPTY_1D

    # Intersection (exact brief test case)
    a = IntervalUnion([(0.0, 1.0), (2.0, 3.0)])
    b = IntervalUnion([(0.5, 2.5)])
    expected_inter = IntervalUnion([(0.5, 1.0), (2.0, 2.5)])
    assert a.intersection(b) == expected_inter
    assert b.intersection(a) == expected_inter

    # Point intersection at boundary (closed intervals)
    c = IntervalUnion([(1.0, 2.0)])
    assert a.intersection(c) == IntervalUnion([(1.0, 1.0), (2.0, 2.0)])
    assert a.intersects(c)

    # Union
    assert a.union(c) == IntervalUnion([(0.0, 3.0)])

    # Subset and intersects
    assert EMPTY_1D.is_subset(a)
    assert not a.is_subset(EMPTY_1D)
    assert not EMPTY_1D.intersects(a)
    assert expected_inter.is_subset(a)
    assert expected_inter.is_subset(b)
    assert not a.is_subset(b)
    assert a.intersects(b)
    assert not a.intersects(IntervalUnion([(1.1, 1.9)]))

    # Tolerance-aware equality (1e-12)
    assert IntervalUnion([(0.0, 1.0)]) == IntervalUnion(
        [(0.0 + 1e-13, 1.0 - 1e-13)]
    )
    assert IntervalUnion([(0.0, 1.0)]) != IntervalUnion([(0.0, 1.0 + 1e-9)])


def test_rect_union_exists_left() -> None:
    r = RectUnion([((0.0, 1.0), (5.0, 6.0)), ((2.0, 3.0), (7.0, 8.0))])
    assert not r.is_empty
    assert RectUnion([]).is_empty

    cond = IntervalUnion([(5.5, 5.7)])
    assert r.exists_left(cond) == IntervalUnion([(0.0, 1.0)])
    assert r.exists_left(EMPTY_1D) == EMPTY_1D
    assert r.exists_left(IntervalUnion([(6.0, 7.0)])) == IntervalUnion(
        [(0.0, 1.0), (2.0, 3.0)]
    )

    # exists_right (symmetric)
    assert r.exists_right(IntervalUnion([(0.5, 0.8)])) == IntervalUnion(
        [(5.0, 6.0)]
    )
    assert r.exists_right(EMPTY_1D) == EMPTY_1D

    # compose (role chains)
    s = RectUnion([((5.5, 6.5), (10.0, 11.0)), ((8.5, 9.0), (12.0, 13.0))])
    comp = r.compose(s)
    assert comp.rects == [((0.0, 1.0), (10.0, 11.0))]
    assert comp.exists_left(IntervalUnion([(10.5, 10.5)])) == IntervalUnion(
        [(0.0, 1.0)]
    )

    # 2-D rectangle union subset
    r_sub = RectUnion([((0.2, 0.8), (5.1, 5.9))])
    assert r_sub.is_subset(r)
    assert not r.is_subset(r_sub)
    r_split = RectUnion(
        [((0.0, 0.5), (5.0, 6.0)), ((0.5, 1.0), (5.0, 6.0)), ((2.0, 3.0), (7.0, 8.0))]
    )
    assert r == r_split


def test_box_algebra() -> None:
    b1 = Box(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
    b2 = Box(np.array([1.0, 1.0]), np.array([3.0, 3.0]))
    b_sub = Box(np.array([0.5, 0.5]), np.array([1.5, 1.5]))
    b_disj = Box(np.array([2.5, 0.0]), np.array([3.5, 1.0]))
    b_emp = Box.empty(2)

    assert not b1.is_empty
    assert b_emp.is_empty
    assert np.all(np.isposinf(b_emp.lo))
    assert np.all(np.isneginf(b_emp.hi))
    assert Box(np.array([2.0, 0.0]), np.array([1.0, 2.0])).is_empty

    # contains (closed)
    assert b1.contains(np.array([0.0, 2.0]))
    assert b1.contains(np.array([1.0, 1.0]))
    assert not b1.contains(np.array([2.1, 1.0]))
    assert not b_emp.contains(np.array([0.0, 0.0]))

    # intersection
    inter = b1.intersection(b2)
    assert not inter.is_empty
    np.testing.assert_allclose(inter.lo, np.array([1.0, 1.0]))
    np.testing.assert_allclose(inter.hi, np.array([2.0, 2.0]))

    assert b1.intersection(b_disj).is_empty
    assert b1.intersection(b_emp).is_empty

    # is_subset & intersects
    assert b_sub.is_subset(b1)
    assert not b1.is_subset(b_sub)
    assert b_emp.is_subset(b1)
    assert b_emp.is_subset(b_emp)
    assert not b1.is_subset(b_emp)

    assert b1.intersects(b2)
    assert not b1.intersects(b_disj)
    assert not b1.intersects(b_emp)

    # Touching boundary intersects
    b_touch = Box(np.array([2.0, 0.0]), np.array([3.0, 1.0]))
    assert b1.intersects(b_touch)
    assert b1.intersection(b_touch).contains(np.array([2.0, 0.5]))


def test_face_algebra() -> None:
    pts4 = simplex_points(4)
    assert pts4.shape == (4, 3)
    np.testing.assert_array_equal(
        pts4,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )

    pts1 = simplex_points(1)
    assert pts1.shape == (1, 0)
    with pytest.raises(ValueError):
        simplex_points(0)

    f01 = Face.from_simplex(pts4, frozenset({0, 1}))
    f12 = Face.from_simplex(pts4, frozenset({1, 2}))
    f0 = Face.from_simplex(pts4, frozenset({0}))
    f_emp = Face.from_simplex(pts4, frozenset())

    assert f01.dim == 3
    assert f01.vertices.shape == (2, 3)
    assert not f01.is_empty
    assert f_emp.is_empty
    assert f_emp.vertices.shape == (0, 3)

    # Intersection and subset on simplex faces
    f1 = f01.intersection(f12)
    assert f1.vertex_ids == frozenset({1})
    np.testing.assert_array_equal(f1.vertices, pts4[[1]])

    assert f0.is_subset(f01)
    assert not f01.is_subset(f0)
    assert f_emp.is_subset(f0)
    assert not f0.is_subset(f_emp)

    # H-representation and contains for face {0, 1} in simplex_points(4)
    A_ub, b_ub, A_eq, b_eq = f01.h_rep()
    assert A_ub.shape == (2, 3)
    assert b_ub.shape == (2,)
    assert A_eq.shape == (2, 3)
    assert b_eq.shape == (2,)

    v0, v1, v2, v3 = pts4[0], pts4[1], pts4[2], pts4[3]
    mid01 = 0.5 * (v0 + v1)
    assert f01.contains(v0)
    assert f01.contains(v1)
    assert f01.contains(mid01)
    assert not f01.contains(v2)
    assert not f01.contains(v3)
    assert not f01.contains(np.array([-0.1, 1.1, 0.0]))
    assert not f_emp.contains(v0)

    # Face containing last vertex (n - 1 = 3, where lambda_3(x) = 1 - sum(x))
    f03 = Face.from_simplex(pts4, frozenset({0, 3}))
    assert f03.contains(v0)
    assert f03.contains(v3)
    assert f03.contains(0.5 * (v0 + v3))
    assert not f03.contains(v1)
    assert not f03.contains(np.array([1.2, 0.0, 0.0]))

    # 0-D simplex face (n = 1, k = 0)
    f_1d_full = Face.from_simplex(pts1, frozenset({0}))
    assert f_1d_full.contains(np.zeros((0,)))
    f_1d_emp = Face.from_simplex(pts1, frozenset())
    assert not f_1d_emp.contains(np.zeros((0,)))

    # Canonical constructor Face(dim, vertices, vertex_ids) matches from_simplex
    f01_direct = Face(3, pts4[[0, 1]], {0, 1})
    assert f01_direct == f01
    np.testing.assert_array_equal(f01_direct.vertices, f01.vertices)
    assert f01_direct.vertex_ids == frozenset({0, 1})


def test_face_validation_and_from_simplex() -> None:
    pts4 = simplex_points(4)  # dim = 3, vertex ids in [0, 3]

    # Face.from_simplex row selection and order normalisation
    f20 = Face.from_simplex(pts4, [2, 0])
    assert f20.dim == 3
    assert f20.vertex_ids == frozenset({0, 2})
    np.testing.assert_array_equal(f20.vertices, pts4[[0, 2]])

    # Validation errors on Face(dim, vertices, vertex_ids)
    # 1. vertices.shape != (len(vertex_ids), dim)
    with pytest.raises(ValueError):
        Face(3, pts4[[0]], {0, 1})
    with pytest.raises(ValueError):
        Face(3, pts4[[0, 1]], set())
    with pytest.raises(ValueError):
        Face(2, pts4[[0, 1]], {0, 1})
    with pytest.raises(ValueError):
        Face(3, np.zeros(3), {0})

    # 2. vertex id out of range [0, dim]
    with pytest.raises(ValueError):
        Face(3, pts4[[0]], {-1})
    with pytest.raises(ValueError):
        Face(3, pts4[[0]], {4})

    # 3. duplicate vertex_ids or negative dim
    with pytest.raises(ValueError):
        Face(3, pts4[[0, 0]], [0, 0])
    with pytest.raises(ValueError):
        Face(-1, np.zeros((0, 0)), set())

    # 4. Validation errors on Face.from_simplex
    with pytest.raises(ValueError):
        Face.from_simplex(pts4, {-1})
    with pytest.raises(ValueError):
        Face.from_simplex(pts4, {4})
    with pytest.raises(ValueError):
        Face.from_simplex(np.zeros((3, 3)), {0})


def test_pair_hull() -> None:
    pts4 = simplex_points(4)
    pairs = np.vstack(
        [
            np.concatenate([pts4[0], pts4[1]]),
            np.concatenate([pts4[2], pts4[3]]),
        ]
    )
    ph = PairHull(pairs)
    assert not ph.is_empty
    assert ph.points.shape == (2, 6)
    np.testing.assert_array_equal(ph.left(), pts4[[0, 2]])
    np.testing.assert_array_equal(ph.right(), pts4[[1, 3]])

    ph_emp = PairHull(np.zeros((0, 6)))
    assert ph_emp.is_empty
    assert ph_emp.left().shape == (0, 3)
    assert ph_emp.right().shape == (0, 3)


def test_import_without_jvm() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mowl.geometric; from mowl.geometric import IntervalUnion, Box, Face",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"


def test_face_rejects_rows_that_disagree_with_vertex_ids() -> None:
    pts4 = simplex_points(4)
    # Rows of vertices 0 and 2 claimed to be vertices 0 and 1.
    with pytest.raises(ValueError, match="canonical simplex vertex"):
        Face(3, pts4[[0, 2]], {0, 1})
    # A non-vertex point claimed to be vertex 3 (the zero vector).
    with pytest.raises(ValueError, match="canonical simplex vertex"):
        Face(3, np.array([[0.25, 0.25, 0.25]]), {3})
    # Rows in sorted-id order are accepted.
    assert Face(3, pts4[[1, 3]], {3, 1}).vertex_ids == frozenset({1, 3})


def test_is_simplex_vertex_rows() -> None:
    from mowl.geometric.regions import is_simplex_vertex_rows

    pts4 = simplex_points(4)
    assert is_simplex_vertex_rows(pts4, 3) is True
    assert is_simplex_vertex_rows(np.zeros((0, 3)), 3) is True
    assert is_simplex_vertex_rows(np.array([[0.5, 0.5, 0.0]]), 3) is False
    assert is_simplex_vertex_rows(np.array([[1.0, 1.0, 0.0]]), 3) is False
    assert is_simplex_vertex_rows(np.array([[2.0, 0.0, 0.0]]), 3) is False
    assert is_simplex_vertex_rows(pts4, 2) is False
