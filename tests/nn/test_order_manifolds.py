"""Distances of the order-embedding manifolds and the disk protrusion at coincident points."""
import pytest
import torch as th

import mowl
mowl.init_jvm("512m")

from mowl.nn.order.disks import protrusion
from mowl.nn.order.manifolds import (Euclidean, Hyperboloid, PoincareBall, Sphere,
                                     UpperHalfSpace)


def points(manifold):
    th.manual_seed(0)
    if isinstance(manifold, Sphere):
        return manifold.project(th.randn(5, 4))
    if isinstance(manifold, PoincareBall):
        return th.rand(5, 4) * 0.3
    if isinstance(manifold, UpperHalfSpace):
        return th.rand(5, 4) + 0.5
    if isinstance(manifold, Hyperboloid):
        return Hyperboloid.from_spatial(th.randn(5, 4))
    return th.randn(5, 4)


MANIFOLDS = [Euclidean, Sphere, PoincareBall, UpperHalfSpace, Hyperboloid]


@pytest.mark.parametrize("cls", MANIFOLDS)
def test_a_disk_inside_itself_costs_exactly_zero(cls):
    m = cls()
    x = points(m)
    r = th.randn(5)
    assert th.equal(m.distance(x, x), th.zeros(5))
    assert th.equal(protrusion(m, x, r, x, r), th.zeros(5))


@pytest.mark.parametrize("cls", MANIFOLDS)
def test_distance_is_unchanged_away_from_coincidence_and_gradients_are_finite(cls):
    m = cls()
    x = points(m).clone().requires_grad_(True)
    y = points(m).roll(1, 0)
    d = m.distance(x, y)
    assert (d > 0.01).all()
    d.sum().backward()
    assert th.isfinite(x.grad).all()
    z = points(m).clone().requires_grad_(True)
    m.distance(z, points(m)).sum().backward()
    assert th.isfinite(z.grad).all()
