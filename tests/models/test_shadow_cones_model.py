"""Shadow cones (Yu et al., ICLR 2024): the four formulations, their relation to entailment
cones (Theorem 3.2), transitivity (Theorem 3.1), the two-case distance, the loss of Eq. 1,
training and ranking."""

import math

import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

import mowl.datasets  # noqa: E402,F401
from mowl.models import ShadowCones  # noqa: E402
from mowl.nn.order import RiemannianSGD, ShadowConesModule  # noqa: E402
from mowl.nn.order.manifolds import PoincareBall, UpperHalfSpace  # noqa: E402
from mowl.representation.library import cone_regions as C  # noqa: E402
from tests.models.order_tree import (common_ranking_benchmark, train_and_separate,  # noqa: E402
                                     tree_dataset)


def _pt(*xs):
    return th.tensor(xs, dtype=th.float64)


def test_penumbral_ball_is_the_entailment_cone():
    """Theorem 3.2: the penumbral cone with a ball light source of radius r at the origin
    has half-aperture arcsin(sinh r (1-|x|^2) / (2|x|)), the entailment cone of Eq. 26 with
    K = sinh(r) / 2, and the same axis angle."""
    r = 0.4
    K = math.sinh(r) / 2
    gen = th.Generator().manual_seed(0)
    for _ in range(20):
        u = th.rand(2, generator=gen, dtype=th.float64) * 1.0 - 0.5
        u = u / u.norm() * (0.3 + 0.6 * th.rand(1, generator=gen).item())
        v = (th.rand(2, generator=gen, dtype=th.float64) - 0.5) * 1.6
        if v.norm() > 0.95:
            continue
        theta = C.OriginLight(r).half_aperture(None, u)
        assert theta.item() == pytest.approx(C.cone_half_aperture(u, K).item(), abs=1e-6)
        e = C.shadow_cone_energy(u, v, "penumbral-ball", radius=r)
        inside_cone = C.entailment_cone_energy(u, v, K).item() < 1e-12
        # the shadow-cone energy is non-positive exactly inside the entailment cone
        assert (e.item() <= 1e-12) == inside_cone


def test_penumbral_signed_distance_is_the_hyperbolic_distance_to_the_boundary():
    """Theorem F.2: for a point on the boundary geodesic the distance is 0, and a point
    on the axis at distance s from the apex is at signed distance asinh(sinh(s) sin(-theta))."""
    u = _pt(0.6, 0.0)
    r = 0.2
    theta = C.OriginLight(r).half_aperture(None, u).item()
    on_axis = _pt(0.9, 0.0)
    s = PoincareBall().distance(u, on_axis).item()
    e = C.shadow_cone_energy(u, on_axis, "penumbral-ball", radius=r).item()
    expected = math.asinh(math.sinh(s) * math.sin(-theta))
    assert e == pytest.approx(expected, abs=1e-5)
    assert e < 0


def test_half_space_distance_and_axis_angle():
    u = _pt(0.3, 0.5)
    below = _pt(0.3, 0.2)
    assert UpperHalfSpace().distance(u, below).item() == pytest.approx(math.log(0.5 / 0.2))
    assert C.HorosphereLight(20.0).axis_angle(None, u, below).item() == pytest.approx(0.0, abs=1e-5)
    above = _pt(0.3, 0.9)
    assert C.HorosphereLight(20.0).axis_angle(None, u, above).item() == pytest.approx(math.pi, abs=1e-5)


@pytest.mark.parametrize("formulation", C.FORMULATIONS)
def test_energy_sign_inside_and_outside_the_cone(formulation):
    if formulation.endswith("half-space"):
        u = _pt(0.0, 0.5)
        inside = _pt(0.0, 0.05)              # far below the apex, on the axis
        outside = _pt(3.0, 0.1)              # far to the side
        above = _pt(0.0, 0.9)                # higher than the apex
    else:
        u = _pt(0.5, 0.0)
        inside = _pt(0.9, 0.0)               # behind the apex, on the axis
        outside = _pt(0.0, 0.7)
        above = _pt(0.3, 0.0)                # between the light source and the apex
    kw = dict(radius=0.1, horosphere_height=20.0)
    assert C.shadow_cone_energy(u, inside, formulation, **kw).item() < 0
    assert C.shadow_cone_energy(u, outside, formulation, **kw).item() > 0
    e_above = C.shadow_cone_energy(u, above, formulation, **kw).item()
    assert e_above > 0
    # for positive altitude the energy is the distance to the apex (paper Sec. 4)
    if formulation.endswith("half-space"):
        d = UpperHalfSpace().distance(u, above).item()
    else:
        d = PoincareBall().distance(u, above).item()
    assert e_above == pytest.approx(d, rel=1e-2, abs=1e-3)


@pytest.mark.parametrize("formulation", ["umbral-half-space", "penumbral-half-space",
                                         "penumbral-ball", "umbral-ball"])
def test_cone_membership_is_transitive(formulation):
    """Theorem 3.1: v in cone(u) and w in cone(v) imply w in cone(u)."""
    gen = th.Generator().manual_seed(1)
    n = 400
    if formulation.endswith("half-space"):
        pts = th.rand(n, 2, generator=gen, dtype=th.float64)
        pts[:, 0] = pts[:, 0] * 1.0 - 0.5
        pts[:, 1] = 0.02 + pts[:, 1] * 0.9
    else:
        d = th.randn(n, 2, generator=gen, dtype=th.float64)
        d = d / d.norm(dim=-1, keepdim=True)
        pts = d * (0.15 + 0.8 * th.rand(n, 1, generator=gen, dtype=th.float64))
    kw = dict(radius=0.2, horosphere_height=20.0, gamma3=0.0)
    energy = C.shadow_cone_energy(pts[:, None, :], pts[None, :, :], formulation, **kw)
    inside = energy <= 0.0
    inside.fill_diagonal_(False)
    chains = (inside.float() @ inside.float()) > 0
    chains.fill_diagonal_(False)
    n_chains = int(chains.sum())
    ok = int((chains & inside).sum())
    assert inside.sum() > 100 and n_chains > 100
    assert ok >= 0.97 * n_chains


def test_umbral_cone_is_inside_the_shadow_of_the_object():
    """An umbral cone is a subset of the shadow of u's ball: its half-space aperture is
    arctan(sinh r), fixed for all u, and wider radius means a wider cone."""
    u = _pt(0.0, 0.5)
    side = _pt(0.3, 0.05)
    narrow = C.shadow_cone_energy(u, side, "umbral-half-space", radius=0.05).item()
    wide = C.shadow_cone_energy(u, side, "umbral-half-space", radius=0.5).item()
    assert wide < narrow


def test_negative_loss_is_a_hinge_and_the_energy_is_selectable():
    """neg=True is max(0, gamma1 - E) (non-negative, zero once pushed out); the raw
    energy of earlier versions is negative=\"energy\"."""
    rows = th.tensor([[0, 1], [2, 1], [3, 4]])
    hinge = ShadowConesModule(6, 3, margin=0.3, seed=0)
    raw = ShadowConesModule(6, 3, margin=0.3, seed=0, negative="energy")
    energy = hinge(rows, "gci0")
    assert th.allclose(raw(rows, "gci0", neg=True), energy)
    out = hinge(rows, "gci0", neg=True)
    assert th.allclose(out, th.relu(0.3 - energy))
    assert (out >= 0).all()
    # a negative deep inside the cone has energy < 0: old "loss" negative, new loss positive
    inside = energy.min().detach().item()
    assert th.relu(th.tensor(0.3 - inside)) > 0


def test_loss_of_eq_1_is_the_same_for_both_forms():
    for form in ("hinge", "energy"):
        model = ShadowCones(tree_dataset(), embed_dim=3, gamma1=0.5, gamma2=0.1, seed=0,
                            negative=form)
        pos = th.tensor([-1.0, 0.3])
        energy = th.tensor([0.2, 1.0, 0.0, 0.1])      # K = 2 blocks of n = 2
        neg = energy if form == "energy" else th.relu(0.5 - energy)
        assert model.training_regime.loss("gci0", pos, neg).item() == pytest.approx(
            _eq1(pos, energy).item())


def _eq1(pos, neg):
    n = neg.reshape(-1, 2).t()
    return th.stack([
        max(pos[i], th.tensor(0.1)) + th.logsumexp(th.relu(0.5 - n[i]), 0)
        for i in range(2)]).mean()


def test_loss_of_eq_1():
    model = ShadowCones(tree_dataset(), embed_dim=3, gamma1=0.5, gamma2=0.1, seed=0,
                        negative="energy")
    pos = th.tensor([-1.0, 0.3])
    neg = th.tensor([0.2, 1.0, 0.0, 0.1])         # K = 2 blocks of n = 2
    n = neg.reshape(-1, 2).t()
    expect = th.stack([
        max(pos[i], th.tensor(0.1)) + th.logsumexp(th.relu(0.5 - n[i]), 0)
        for i in range(2)]).mean()
    assert model.training_regime.loss("gci0", pos, neg).item() == pytest.approx(expect.item())


def test_half_space_update_keeps_positive_height_and_ball_update_avoids_the_hole():
    for formulation in ("umbral-half-space", "umbral-ball"):
        m = ShadowConesModule(10, 3, formulation=formulation, radius=0.1, seed=0)
        opt = RiemannianSGD(m.param_groups(), lr=50.0)
        rows = th.tensor([[i, (i + 3) % 10] for i in range(10)])
        m(rows, "gci0").sum().backward()
        opt.step()
        w = m.class_embed.weight
        assert th.isfinite(w).all()
        if formulation.endswith("half-space"):
            assert (w[:, -1] > 0).all()
        else:
            assert w.norm(dim=-1).max() < 1
            assert w.norm(dim=-1).min() >= math.tanh(0.05) - 1e-6   # outside the hole


@pytest.mark.parametrize("formulation,lr", [("umbral-half-space", 0.01),
                                            ("penumbral-half-space", 0.01),
                                            ("penumbral-ball", 0.05)])
def test_training_separates_ancestors_from_other_classes(formulation, lr):
    model = ShadowCones(tree_dataset(), embed_dim=5, formulation=formulation,
                        learning_rate=lr, seed=0)
    pos, neg = train_and_separate(model, epochs=300)
    assert pos < neg - 0.5 * abs(neg)


def test_code_loss_regime_can_be_attached():
    """The authors' code trains with a softmax cross-entropy; mOWL has it as a regime."""
    from mowl.training import SoftmaxEnergyContrastiveLoss
    model = ShadowCones(tree_dataset(), embed_dim=3, seed=0)
    model.training_regime = SoftmaxEnergyContrastiveLoss(gamma2=0.001)
    model.train(epochs=3)
    assert th.isfinite(model.module.class_embed.weight).all()


def test_unknown_formulation_and_forms():
    with pytest.raises(ValueError):
        ShadowConesModule(3, 3, formulation="nope")
    m = ShadowConesModule(3, 3)
    with pytest.raises(NotImplementedError):
        m(th.tensor([[0, 1, 2]]), "gci2")


def test_ranked_by_the_joint_subsumption_benchmark():
    from tests.benchmarks.tiny import scratch_dir, tiny_dataset
    from mowl.benchmarks.comparison import compare_methods
    outcome = compare_methods(common_ranking_benchmark(), methods=["shadow_cones"],
                              epochs=3, seeds=(0,), model_kwargs={"embed_dim": 4},
                              dataset=tiny_dataset(), workdir=scratch_dir(),
                              stop_on_error=True)
    (result,) = outcome.results
    assert 0.0 < result.metrics["all"]["f_mrr"] <= 1.0
