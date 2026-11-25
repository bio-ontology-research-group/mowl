"""The library's cone energies (regions and lights of ``cone_regions``) against the frozen
direct functions of ``mowl.baselines.nn.order.cones``: two independent implementations of the
shadow cones (Yu et al. 2024) and entailment cones (Ganea et al. 2018) agree on random points;
only the published clamps (arccosine of the cosine clamped to 1 - 1e-6) differ, and only for
parent and child on one ray."""
import pytest
import torch as th

import mowl

mowl.init_jvm("2g")

from mowl.baselines.nn.order import cones as B  # noqa: E402
from mowl.representation.library import cone_regions as C  # noqa: E402


def _points(form, n, seed):
    g = th.Generator().manual_seed(seed)
    if form.endswith("half-space"):
        return th.cat([th.randn(n, 2, generator=g) * 0.5,
                       th.exp(th.randn(n, 1, generator=g) * 0.7)], dim=1)
    b = th.randn(n, 2, generator=g)
    return b / b.norm(dim=1, keepdim=True) * (0.05 + 0.9 * th.rand(n, 1, generator=g))


@pytest.mark.parametrize("form", C.FORMULATIONS)
def test_library_energy_equals_the_direct_one(form):
    u, v = _points(form, 20000, 1).double(), _points(form, 20000, 2).double()
    new = C.shadow_cone_energy(u, v, form, 0.05, 20.0, 1e-4)
    old = B.shadow_cone_energy(u, v, form, 0.05, 20.0, 1e-4)
    # the published clamp of the arccosine (to 1 - 1e-6) matters for parent and child on one
    # ray only: the angle at the apex (at the origin for the umbral ball) is within 1.4e-3 of
    # 0 or pi there
    if form == "umbral-ball":
        angle = C._angle_between(u, v)
    elif form == "penumbral-ball":
        angle = C.axis_angle(u, v)
    else:
        angle = C.HorosphereLight(20.0).axis_angle(None, u, v)
    keep = th.sin(angle).abs() > 3e-3
    assert int(keep.sum()) > 19000
    assert float((new - old).abs()[keep].max()) < 1e-5


@pytest.mark.parametrize("K", [0.1, 0.3])
def test_entailment_cone_equals_the_direct_one(K):
    u, v = _points("ball", 20000, 3).double(), _points("ball", 20000, 4).double()
    assert float((C.cone_half_aperture(u, K) - B.cone_half_aperture(u, K)).abs().max()) < 1e-9
    keep = th.sin(C.axis_angle(u, v)).abs() > 3e-3      # the published clamp: one ray only
    assert float((C.entailment_cone_energy(u, v, K) - B.entailment_cone_energy(u, v, K)
                  ).abs()[keep].max()) < 1e-5
    assert C.entailment_cone_min_norm(K) == B.entailment_cone_min_norm(K)


@pytest.mark.parametrize("form", C.FORMULATIONS)
def test_a_point_costs_nothing_in_its_own_cone(form):
    u = _points(form, 20000, 3)
    assert float(C.shadow_cone_energy(u, u.clone(), form, 0.05, 20.0, 1e-4).abs().max()) == 0.0
