"""Hyperbolic entailment cones (Ganea et al., ICML 2018): geometry of the energy, the
transitivity theorem, the Riemannian update, and training and ranking end to end."""

import math

import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

import mowl.datasets  # noqa: E402,F401
from mowl.models import HyperbolicCones  # noqa: E402
from mowl.nn.order import HyperbolicConesModule, RiemannianSGD  # noqa: E402
from mowl.nn.order import cones as C  # noqa: E402
from mowl.nn.order.manifolds import PoincareBall  # noqa: E402
from tests.models.order_tree import (common_ranking_benchmark,  # noqa: E402
                                     train_and_separate, tree_dataset)

K = 0.1


def _pt(*xs):
    return th.tensor(xs, dtype=th.float64)


def test_aperture_is_the_paper_formula_and_below_half_pi():
    x = _pt(0.5, 0.0)
    assert C.cone_half_aperture(x, K).item() == pytest.approx(math.asin(K * (1 - 0.25) / 0.5))
    eps = C.entailment_cone_min_norm(K)
    assert eps == pytest.approx(2 * K / (1 + math.sqrt(1 + 4 * K * K)))
    # at the boundary of the epsilon-hole the aperture is exactly pi / 2 (Eq. 25)
    assert C.cone_half_aperture(_pt(eps, 0.0), K).item() == pytest.approx(math.pi / 2, abs=5e-3)
    for r in (0.3, 0.6, 0.9):
        assert 0 < C.cone_half_aperture(_pt(r, 0.0), K).item() < math.pi / 2


def test_energy_is_zero_inside_the_cone_and_positive_outside():
    u = _pt(0.5, 0.0)
    on_axis = _pt(0.8, 0.0)          # further out along the axis: inside
    towards_origin = _pt(0.3, 0.0)   # on the other side of u: angle pi
    sideways = _pt(0.5, 0.4)
    assert C.entailment_cone_energy(u, on_axis, K).item() == pytest.approx(0.0, abs=1e-5)
    assert C.axis_angle(u, on_axis).item() == pytest.approx(0.0, abs=1e-2)
    assert C.axis_angle(u, towards_origin).item() == pytest.approx(math.pi, abs=1e-2)
    assert C.entailment_cone_energy(u, towards_origin, K).item() > 1.0
    assert C.entailment_cone_energy(u, sideways, K).item() > 0.0


def test_energy_equals_the_angle_to_the_cone_boundary():
    """E is the smallest rotation about u that brings v into the cone: for a point at
    angle Xi the energy is Xi - psi."""
    u = _pt(0.0, 0.6)
    v = _pt(0.3, 0.6)
    xi = C.axis_angle(u, v)
    psi = C.cone_half_aperture(u, K)
    assert C.entailment_cone_energy(u, v, K).item() == pytest.approx(max(0.0, (xi - psi).item()))


def test_cone_relation_is_transitive():
    """Theorem 4: with psi of Eq. 26, v in cone(u) and w in cone(v) give w in cone(u)."""
    gen = th.Generator().manual_seed(0)
    eps = C.entailment_cone_min_norm(K)

    def sample(n):
        d = th.randn(n, 2, generator=gen, dtype=th.float64)
        d = d / d.norm(dim=-1, keepdim=True)
        return d * (eps + (0.95 - eps) * th.rand(n, 1, generator=gen, dtype=th.float64))

    pts = sample(120)
    inside = lambda u, v: C.entailment_cone_energy(u, v, K).item() < 1e-9  # noqa: E731
    chains = checked = 0
    for u in pts:
        kids = [v for v in pts if inside(u, v) and (v - u).norm() > 1e-6]
        for v in kids:
            for w in pts:
                if (w - v).norm() > 1e-6 and inside(v, w):
                    chains += 1
                    checked += inside(u, w)
    assert chains > 100
    # a few boundary cases fail by floating point tolerance only
    assert checked >= 0.99 * chains


def test_negative_pairs_use_the_margin():
    m = HyperbolicConesModule(4, 3, K=K, margin=0.5, seed=0)
    rows = th.tensor([[0, 1], [2, 3]])
    e = m(rows, "gci0")
    assert th.allclose(m(rows, "gci0", neg=True), th.relu(0.5 - e))
    with pytest.raises(NotImplementedError):
        m(th.tensor([[0, 1, 2]]), "gci2")


def test_initial_points_respect_the_hole_and_the_ball():
    m = HyperbolicConesModule(50, 5, K=K, seed=1)
    n = m.class_embed.weight.norm(dim=-1)
    assert n.min() >= m.epsilon - 1e-6 and n.max() < 1


def test_rsgd_step_stays_in_the_ball_and_outside_the_hole():
    m = HyperbolicConesModule(10, 3, K=K, seed=0)
    opt = RiemannianSGD(m.param_groups(), lr=10.0)  # a huge step
    rows = th.tensor([[i, (i + 1) % 10] for i in range(10)])
    m(rows, "gci0").sum().backward()
    opt.step()
    n = m.class_embed.weight.norm(dim=-1)
    assert th.isfinite(m.class_embed.weight).all()
    assert n.max() < 1 and n.min() >= m.epsilon - 1e-6
    # the Riemannian gradient is the Euclidean one scaled by (1 - |x|^2)^2 / 4
    x = _pt(0.5, 0.0)
    g = _pt(1.0, 1.0)
    assert th.allclose(PoincareBall().rgrad(x, g), g * (0.75 ** 2 / 4))


def test_training_separates_ancestors_from_other_classes():
    model = HyperbolicCones(tree_dataset(), embed_dim=5, seed=0)
    assert len(model.training_datasets["gci0"]) == 15
    pos, neg = train_and_separate(model, epochs=300)
    assert pos < 0.5 * neg


def test_only_subclass_axioms_between_named_classes_are_used():
    n = "http://mowl.borg/order#"
    extra = [f"SubClassOf(<{n}A> ObjectSomeValuesFrom(<{n}r> <{n}B>))",
             f"SubClassOf(ObjectIntersectionOf(<{n}A> <{n}B>) <{n}R>)",
             f"SubClassOf(<{n}A> <{n}A>)"]
    model = HyperbolicCones(tree_dataset(extra), embed_dim=3, seed=0)
    rows = model.training_datasets["gci0"][:]
    n_named = len(model.named_class_index_dict)
    assert (rows < n_named).all() and (rows[:, 0] != rows[:, 1]).all()
    assert len(rows) == 15
    assert model.scored_forms == ("gci0",) and model.axiom_scoring == "module"


def test_negatives_are_not_told_axioms_and_use_both_sides():
    import numpy as np
    np.random.seed(0)
    model = HyperbolicCones(tree_dataset(), embed_dim=3, num_neg=4, seed=0)
    ds = model.training_datasets["gci0"]
    neg = model.generate_negatives("gci0", ds)
    assert neg.shape == (4 * len(ds), 2)
    told = {tuple(r) for r in ds[:].tolist()}
    assert not any(tuple(r) in told for r in neg.tolist())
    pos = ds[:]
    n = len(ds)
    assert (neg[:n, 1] == pos[:, 1]).all()           # first block corrupts the child
    assert (neg[n:2 * n, 0] == pos[:, 0]).all()      # second block corrupts the parent
    assert (neg[2 * n:3 * n, 1] == pos[:, 1]).all()  # third block corrupts the child again


def test_ranked_by_the_joint_subsumption_benchmark():
    from tests.benchmarks.tiny import scratch_dir, tiny_dataset
    from mowl.benchmarks.comparison import compare_methods
    outcome = compare_methods(common_ranking_benchmark(), methods=["hyperbolic_cones"],
                              epochs=3, seeds=(0,), model_kwargs={"embed_dim": 4},
                              dataset=tiny_dataset(), workdir=scratch_dir(),
                              stop_on_error=True)
    (result,) = outcome.results
    assert 0.0 < result.metrics["all"]["f_mrr"] <= 1.0
    assert result.protocol_info["scoring"] == "loss"
