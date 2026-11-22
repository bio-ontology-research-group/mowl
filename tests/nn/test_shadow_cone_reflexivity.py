"""Shadow cone energies: coincident points cost exactly 0, everything else is unchanged."""
import math

import pytest
import torch as th

import mowl

mowl.init_jvm("2g")

from mowl.nn.lm import hyperbolic as H  # noqa: E402
from mowl.nn.order import cones as C  # noqa: E402


def _points(form, n, seed):
    g = th.Generator().manual_seed(seed)
    if form.endswith("half-space"):
        return th.cat([th.randn(n, 2, generator=g) * 0.5,
                       th.exp(th.randn(n, 1, generator=g) * 0.7)], dim=1)
    b = th.randn(n, 2, generator=g)
    return b / b.norm(dim=1, keepdim=True) * (0.05 + 0.9 * th.rand(n, 1, generator=g))


def _old_half_space_distance(x, y):
    diff = ((x - y) ** 2).sum(-1)
    return th.acosh((1 + diff / (2 * x[..., -1] * y[..., -1])).clamp_min(1.0 + 1e-7))


def _old_energy(u, v, form, radius=0.05, height=20.0, gamma3=1e-4):
    """The energies as they were before the fix (clamped acos and acosh), with the two
    changed functions swapped back in."""
    if form == "umbral-ball":
        nu, nv = (u * u).sum(-1), (v * v).sum(-1)
        cos = (u * v).sum(-1) / (nu.sqrt() * nv.sqrt())
        alpha = th.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        sinh_du = 2 * nu.sqrt() / (1 - nu)
        sinh_dv = 2 * nv.sqrt() / (1 - nv)
        beta = th.asin((math.sinh(radius) / sinh_du).clamp(max=1 - 1e-6))
        t = sinh_dv * th.sin(alpha - beta)
        signed = th.asinh(t) + radius
        h_v = th.acosh((th.sqrt(1 + sinh_dv ** 2) / th.sqrt(1 + t ** 2)).clamp_min(1.0))
        h_u = th.acosh((th.sqrt(1 + sinh_du ** 2) / math.cosh(radius)).clamp_min(1.0))
        up = C._push_outward_ball(u, gamma3)
        return th.where(h_u - h_v > 0, H.distance(up, v, 1.0), signed)
    saved = C.half_space_distance
    C.half_space_distance = _old_half_space_distance
    try:
        return C._shadow_cone_energy(u, v, form, radius, height, gamma3)
    finally:
        C.half_space_distance = saved


@pytest.mark.parametrize("form", C.FORMULATIONS)
def test_non_coincident_points_keep_their_energy(form):
    # in float64, so that the rounding of the old acos near 1 (amplified by the cone) is
    # not what is compared; the pairs whose parent and child lie on the same ray to within
    # the old clamp (angle < 1.4e-3) are the ones the fix changes
    u, v = _points(form, 20000, 1).double(), _points(form, 20000, 2).double()
    new = C.shadow_cone_energy(u, v, form, 0.05, 20.0, 1e-4)
    old = _old_energy(u, v, form)
    keep = th.ones_like(new, dtype=th.bool)
    if form == "umbral-ball":
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
        keep = cos < 1 - 1e-6
    assert int(keep.sum()) > 19000
    assert float((new - old).abs()[keep].max()) < 1e-5


@pytest.mark.parametrize("form", C.FORMULATIONS)
def test_a_point_costs_nothing_in_its_own_cone(form):
    u = _points(form, 20000, 3)
    assert float(C.shadow_cone_energy(u, u.clone(), form, 0.05, 20.0, 1e-4).abs().max()) == 0.0
