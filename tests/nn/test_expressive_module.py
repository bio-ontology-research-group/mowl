"""Geometry of ExpressivE: what the paper states about the hyper-parallelogram."""
import torch as th

import mowl
mowl.init_jvm("1g")
import mowl.datasets  # noqa: F401,E402
from mowl.nn.graph.expressive import (ExpressivEInteraction, captured,  # noqa: E402
                                      expressive_distance, expressive_relation_parameters)

D = 3


def _rel(d_h, d_t, c_h, c_t, s_h, s_t):
    """Relation parameters already in the constrained form, as tensors of size ``(1, D)``."""
    f = lambda v: th.full((1, D), float(v))  # noqa: E731
    return tuple(f(v) for v in (d_h, d_t, c_h, c_t, s_h, s_t))


def _in_parallelogram(h, t, rel):
    """The two inequalities of the paper, written out per dimension."""
    d_h, d_t, c_h, c_t, s_h, s_t = rel
    first = (h - c_h - s_t * t).abs() <= d_h
    second = (t - c_t - s_h * h).abs() <= d_t
    return bool((first & second).all())


def test_captured_agrees_with_the_inequalities_of_the_paper():
    th.manual_seed(0)
    rel = _rel(0.3, 0.4, 0.1, -0.2, 0.5, -0.3)
    for _ in range(200):
        h, t = th.rand(1, D) * 2 - 1, th.rand(1, D) * 2 - 1
        assert bool(captured(h, t, *rel)) == _in_parallelogram(h, t, rel)


def test_distance_is_small_inside_and_large_outside():
    rel = _rel(0.3, 0.3, 0.0, 0.0, 0.0, 0.0)
    inside = expressive_distance(th.full((1, D), 0.1), th.full((1, D), -0.1), *rel)
    outside = expressive_distance(th.full((1, D), 0.9), th.full((1, D), -0.9), *rel)
    assert inside.norm() < outside.norm()
    # inside the parallelogram the distance is |tau| / w <= d / w < d
    assert (inside <= 0.3).all()


def test_symmetry_pattern():
    """Equal head and tail parameters give a symmetric relation (Theorem 5.2)."""
    th.manual_seed(1)
    rel = _rel(0.3, 0.3, 0.1, 0.1, 0.4, 0.4)
    h, t = th.rand(5, D), th.rand(5, D)
    # the head and tail halves of the distance vector swap; the score (a norm) is equal
    assert th.allclose(expressive_distance(h, t, *rel).norm(dim=-1),
                       expressive_distance(t, h, *rel).norm(dim=-1))
    assert bool((captured(h, t, *rel) == captured(t, h, *rel)).all())


def test_hierarchy_pattern():
    """A parallelogram contained in another captures a sub-relation: every pair inside the
    first is inside the second (Theorem 5.3, hierarchy)."""
    th.manual_seed(2)
    small = _rel(0.15, 0.15, 0.05, 0.05, 0.3, 0.2)
    large = _rel(0.4, 0.4, 0.05, 0.05, 0.3, 0.2)
    found = 0
    for _ in range(2000):
        h, t = th.rand(1, D) * 0.4 - 0.2, th.rand(1, D) * 0.4 - 0.2
        if bool(captured(h, t, *small)):
            found += 1
            assert bool(captured(h, t, *large))
    assert found > 0


def test_mutual_exclusion_pattern():
    """Parallelograms with disjoint centres and small widths share no pair."""
    th.manual_seed(3)
    a = _rel(0.1, 0.1, 0.6, 0.6, 0.0, 0.0)
    b = _rel(0.1, 0.1, -0.6, -0.6, 0.0, 0.0)
    for _ in range(500):
        h, t = th.rand(1, D) * 2 - 1, th.rand(1, D) * 2 - 1
        assert not (bool(captured(h, t, *a)) and bool(captured(h, t, *b)))


def test_relation_constraints_of_the_reference_code():
    raw = th.randn(7, 6 * D) * 3
    d_h, d_t, c_h, c_t, s_h, s_t = expressive_relation_parameters(raw, True, 0.5)
    assert (d_h >= 0).all() and (d_t >= 0).all() and (d_h <= 1).all()
    assert (c_h.abs() <= 1).all()
    assert ((1 - s_h * s_t).abs() >= 0.5 - 1e-4).all()


def test_interaction_score_is_the_negative_norm_of_the_distance():
    th.manual_seed(4)
    interaction = ExpressivEInteraction(p=2, tanh_map=True, min_denom=0.5)
    h, t, r = th.randn(4, D), th.randn(4, D), th.randn(4, 6 * D)
    params = expressive_relation_parameters(r, True, 0.5)
    expected = -expressive_distance(th.tanh(h), th.tanh(t), *params).norm(dim=-1)
    assert th.allclose(interaction(h, r, t), expected)


def test_a_captured_pair_scores_above_an_uncaptured_one():
    interaction = ExpressivEInteraction(p=2, tanh_map=False, min_denom=0.0)
    r = th.cat([th.full((1, D), 0.3)] * 2 + [th.zeros(1, D)] * 4, dim=-1)
    near = interaction(th.full((1, D), 0.1), r, th.full((1, D), 0.1))
    far = interaction(th.full((1, D), 0.8), r, th.full((1, D), -0.8))
    assert near.item() > far.item()


def test_compositional_definition_pattern():
    """r1(X, Y) & r2(Y, Z) <=> rd(X, Z) (Theorem 5.4): relations that are functions, as
    zero-width parallelograms (slopes a, centres b with y = a x + b), compose like the
    functions do. Parameters are used in their constrained form, without ``min_denom``."""
    th.manual_seed(5)
    eps = 1e-5

    def function(a, b):
        # y = a x + b: |t - c^t - s^h h| <= 0 with c^t = b, s^h = a, and the head
        # inequality |h - c^h - s^t t| <= 0 with s^t = 1 / a, c^h = -b / a
        return _rel(eps, eps, -b / a, b, a, 1 / a)

    a1, b1, a2, b2 = 0.5, 0.1, -0.4, 0.2
    r1, r2 = function(a1, b1), function(a2, b2)
    rd = function(a2 * a1, a2 * b1 + b2)
    for _ in range(100):
        x = th.rand(1, D) * 0.6 - 0.3
        y = a1 * x + b1
        z = a2 * y + b2
        assert bool(captured(x, y, *r1)) and bool(captured(y, z, *r2))
        assert bool(captured(x, z, *rd))
        # a different third entity is not in the relation
        assert not bool(captured(x, z + 0.3, *rd))
