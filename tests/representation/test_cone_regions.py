"""The cones of ``cone_regions`` against brute-force geometry: the boundary of a shadow cone is
found by sampling geodesics and hyperbolic balls, and the distance of a point to it is the
minimum over the samples. This is the check against the definitions of Yu et al. (2024,
Section 4), independent of the closed forms the library uses (and of the direct module of
``mowl.baselines``)."""
import math

import pytest
import torch as th

import mowl

mowl.init_jvm("2g")

from mowl.nn.lm import hyperbolic as H  # noqa: E402
from mowl.nn.order.manifolds import PoincareBall, UpperHalfSpace  # noqa: E402
from mowl.representation.library import cone_regions as C  # noqa: E402
from mowl.representation.library.hierarchy_regions import Point  # noqa: E402

D = th.float64
S = th.linspace(-12.0, 12.0, 24001, dtype=D)           # arclength grid of a geodesic


def _geodesic(a, b):
    """Points of the geodesic through ``a`` and ``b`` of the Poincare disk, ``(len(S), 2)``."""
    direction = H.mobius_add(-a, b, 1.0)
    direction = direction / direction.norm()
    return H.mobius_add(a.expand(len(S), 2), th.tanh(S / 2).unsqueeze(-1) * direction, 1.0)


def _dist(x, y):
    return H.distance(x, y, 1.0)


def _tangent_line_through_origin(u, r):
    """The geodesic through O tangent to the circle of radius r around u (the boundary of the
    umbra of B(u, r) lit from O), on the side of positive angle: by bisection on the angle."""
    def gap(omega):
        line = th.tanh(S / 2).unsqueeze(-1) * th.stack([th.cos(omega), th.sin(omega)])
        return _dist(line, u.expand(len(S), 2)).min() - r
    base = th.atan2(u[1], u[0])
    lo, hi = base, base + math.pi / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    omega = (lo + hi) / 2
    return th.stack([th.cos(omega), th.sin(omega)])


@pytest.mark.parametrize("u,v", [((0.55, 0.1), (0.8, 0.5)), ((0.6, 0.0), (0.2, 0.7)),
                                 ((0.5, -0.2), (0.85, -0.5))])
def test_umbral_ball_is_the_signed_distance_to_the_tangent_line_plus_the_radius(u, v):
    r = 0.2
    u, v = th.tensor(u, dtype=D), th.tensor(v, dtype=D)
    direction = _tangent_line_through_origin(u, r)
    # the cone is symmetric about the ray through u: the other wall is the mirror image
    base = th.atan2(u[1], u[0])
    omega = 2 * base - th.atan2(direction[1], direction[0])
    mirror = th.stack([th.cos(omega), th.sin(omega)])
    brute = min(float(_dist(th.tanh(S / 2).unsqueeze(-1) * d, v.expand(len(S), 2)).min())
                for d in (direction, mirror))
    t, closer = C.OriginLight(r).umbra(PoincareBall(), u[None], v[None], r)
    assert abs(float(th.asinh(t).abs()) - float(brute)) < 1e-3
    del closer


def test_the_aperture_of_a_light_ball_is_the_tangent_angle():
    """The cone of a point apex lit by a ball of radius r at the origin opens by the angle at
    which the geodesics through the apex are tangent to the ball (found by sampling)."""
    r = 0.15
    u = th.tensor([0.6, 0.0], dtype=D)
    point_ball = th.tensor([0.6 + 1e-5, 0.0], dtype=D)

    def gap(psi):
        b = u + 1e-6 * th.stack([th.cos(psi), th.sin(psi)])
        line = _geodesic(u, b)
        return _dist(line, th.zeros(2, dtype=D).expand(len(S), 2)).min() - r
    lo, hi = th.tensor(math.pi / 2, dtype=D), th.tensor(math.pi, dtype=D)  # towards the origin
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if gap(mid) < 0 else (mid, hi)
    psi = (lo + hi) / 2
    axis_to_boundary = math.pi - float(psi)               # the axis points away from O (+x)
    theta = float(C.OriginLight(r).half_aperture(PoincareBall(), u[None]))
    assert axis_to_boundary == pytest.approx(theta, abs=2e-3)
    del point_ball


def test_the_umbra_in_the_half_space_is_the_cylinder_of_the_ball():
    """The shadow of the hyperbolic ball B(u, r) lit from infinity has the horizontal
    half-width u_n sinh r (sampled: the widest horizontal extent of the ball)."""
    r, un = 0.3, 0.7
    u = th.tensor([0.0, un], dtype=D)
    hs = UpperHalfSpace()
    xs = th.linspace(0.0, 1.5, 3001, dtype=D)
    ys = th.exp(th.linspace(-3.0, 3.0, 6001, dtype=D))
    grid = th.stack(th.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    inside = hs.distance(u.expand(len(grid), 2), grid) <= r
    width = float(grid[inside][:, 0].max())
    assert width == pytest.approx(un * math.sinh(r), abs=1e-3)
    # a point on the wall of the cylinder has distance exactly r - r = 0 to the cone
    wall = th.tensor([[un * math.sinh(r), 0.1]], dtype=D)
    t, _ = C.InfinityLight().umbra(hs, u[None], wall, r)
    assert float(t) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("form", C.FORMULATIONS)
def test_the_cone_of_a_point_apex_contains_the_apex_and_its_axis(form):
    light, umbral = C.light_for(form)
    ball = form.endswith("ball")
    manifold = PoincareBall() if ball else UpperHalfSpace()
    apex = th.tensor([[0.5, 0.0]] if ball else [[0.0, 0.5]], dtype=D)
    below = th.tensor([[0.9, 0.0]] if ball else [[0.0, 0.05]], dtype=D)   # away from the light
    shadow = C.Shadow(apex, manifold, None, light, umbral, 0.05, 1e-4)
    assert float(shadow.signed_distance(apex)) == 0.0
    assert float(shadow.signed_distance(below)) < 0.0


def test_entailment_cone_is_the_penumbral_cone_of_a_light_of_radius_asinh_2K():
    K = 0.1
    x = th.tensor([[0.4, 0.3], [0.1, -0.6]], dtype=D)
    light = C.OriginLight(math.asinh(2 * K))
    assert th.allclose(C.cone_half_aperture(x, K), light.half_aperture(None, x))
    n = x.norm(dim=-1)
    assert th.allclose(C.cone_half_aperture(x, K), th.asin(K * (1 - n * n) / n))


def test_a_light_refuses_the_wrong_manifold():
    shadow = C.Shadow(th.zeros(1, 2), UpperHalfSpace(), None, C.OriginLight(0.1), True)
    with pytest.raises(Exception):
        shadow.signed_distance(th.zeros(1, 2) + 0.5)
    assert isinstance(Point(th.zeros(1, 2)), Point)


def test_umbral_and_penumbral_ball_agree_where_the_apex_branch_applies():
    """Both ball formulations are the distance to the (pushed) apex for a point on the light
    side of the apex, as Section 4 defines it; they differ elsewhere (different boundary
    lines). This is the observation that the two energies coincide in the apex branch."""
    r = 0.1
    u = th.tensor([[0.6, 0.0]], dtype=D)
    v = th.tensor([[0.3, 0.05]], dtype=D)           # between the light and the apex
    a = C.shadow_cone_energy(u, v, "umbral-ball", r)
    b = C.shadow_cone_energy(u, v, "penumbral-ball", r)
    up = C.OriginLight(r).push(None, u, 1e-4)
    assert float(a) == pytest.approx(float(b), abs=1e-12)
    assert float(a) == pytest.approx(float(PoincareBall().distance(up, v)), abs=1e-12)
    w = th.tensor([[0.9, 0.4]], dtype=D)            # behind the apex, off the axis
    assert abs(float(C.shadow_cone_energy(u, w, "umbral-ball", r)
                     - C.shadow_cone_energy(u, w, "penumbral-ball", r))) > 1e-3
