"""Disk embeddings (Suzuki et al., ICML 2019): protrusion, inclusion, the manifolds and the
Riemannian update, the margin loss, training and ranking."""

import math

import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

import mowl.datasets  # noqa: E402,F401
from mowl.models import DiskEmbeddings  # noqa: E402
from mowl.nn.order import DiskEmbeddingsModule, RiemannianSGD  # noqa: E402
from mowl.baselines.nn.order.disks import protrusion  # noqa: E402
from mowl.nn.order.manifolds import Euclidean, Hyperboloid, Sphere  # noqa: E402
from tests.models.order_tree import (common_ranking_benchmark, train_and_separate,  # noqa: E402
                                     tree_dataset)


def _hyp(*spatial):
    return Hyperboloid.from_spatial(th.tensor(spatial, dtype=th.float64))


def test_hyperboloid_points_and_distance():
    m = Hyperboloid()
    origin = _hyp(0.0, 0.0)
    p = _hyp(math.sinh(1.3), 0.0)
    assert (-th.sum(p[1:] ** 2) + p[0] ** 2).item() == pytest.approx(1.0)
    assert m.distance(origin, p).item() == pytest.approx(1.3, abs=1e-6)
    q = _hyp(0.4, -0.7)
    assert m.distance(p, q).item() == pytest.approx(m.distance(q, p).item())


def test_protrusion_is_nonpositive_exactly_when_the_disk_contains_the_other():
    m = Euclidean()
    big = (th.tensor([0.0, 0.0]), th.tensor(2.0))
    inside = (th.tensor([1.0, 0.0]), th.tensor(0.5))      # d = 1 <= 2 - 0.5
    touching = (th.tensor([1.5, 0.0]), th.tensor(0.5))    # d = 1.5 = 2 - 0.5
    outside = (th.tensor([1.6, 0.0]), th.tensor(0.5))
    prot = lambda small: protrusion(m, big[0], big[1], small[0], small[1]).item()  # noqa: E731
    assert prot(inside) < 0
    assert prot(touching) == pytest.approx(0.0, abs=1e-6)
    assert prot(outside) > 0
    # a disk with a negative radius is allowed (Fig. 1)
    neg = (th.tensor([0.5, 0.0]), th.tensor(-0.2))
    assert prot(neg) < 0


@pytest.mark.parametrize("geometry", ["hyperbolic", "spherical", "euclidean"])
def test_inclusion_is_transitive_and_shift_invariant(geometry):
    """Transitivity follows from the triangle inequality; adding a constant to every
    radius changes no protrusion (Eq. 4)."""
    mod = DiskEmbeddingsModule(40, 3, geometry=geometry, init_scale=0.6, seed=0)
    with th.no_grad():
        mod.radius_embed.weight.copy_(th.randn(40, 1, generator=th.Generator().manual_seed(2)))
    idx = th.arange(40)
    pair = lambda a, b: mod.energy(b, a)  # noqa: E731  (a contains b: l(a, b) <= 0)
    L = th.stack([pair(idx[i].repeat(40), idx) for i in range(40)])     # L[i, j] = l(i, j)
    inside = L <= 0
    inside.fill_diagonal_(False)
    chains = (inside.float() @ inside.float()) > 0
    chains.fill_diagonal_(False)
    assert int(chains.sum()) > 20
    assert bool((chains & (L <= 1e-5)).sum() == chains.sum())
    with th.no_grad():
        mod.radius_embed.weight += 3.0
    L2 = th.stack([pair(idx[i].repeat(40), idx) for i in range(40)])
    assert th.allclose(L, L2, atol=1e-5)


def test_reversibility_of_the_metric_case():
    """Eq. 5: (x, r) contains (y, s) iff (x, -r) is contained in (y, -s), for a symmetric d."""
    m = Hyperboloid()
    x, y = _hyp(0.3, 0.1), _hyp(-0.5, 0.2)
    r, s = 2.0, 0.5
    contains = protrusion(m, x, r, y, s).item() <= 0
    # l(y, -s ; x, -r) = d - (-s) + (-r) = d + s - r, the same number as l(x, r ; y, s)
    flipped = protrusion(m, y, -s, x, -r).item() <= 0
    assert contains == flipped


def test_rsgd_steps_stay_on_the_manifold():
    for geometry in ("hyperbolic", "spherical"):
        mod = DiskEmbeddingsModule(8, 3, geometry=geometry, seed=0)
        opt = RiemannianSGD(mod.param_groups(), lr=5.0)
        rows = th.tensor([[i, (i + 1) % 8] for i in range(8)])
        (mod(rows, "gci0") ** 2).sum().backward()
        opt.step()
        w = mod.class_embed.weight
        assert th.isfinite(w).all()
        if geometry == "hyperbolic":
            inner = -(w[:, 0] ** 2) + (w[:, 1:] ** 2).sum(-1)
            assert th.allclose(inner, -th.ones(8), atol=1e-3)
        else:
            assert th.allclose(w.norm(dim=-1), th.ones(8), atol=1e-5)
    # the Riemannian gradient on the hyperboloid is tangent: <x, grad>_L = 0
    x = _hyp(0.3, -0.2)
    g = th.tensor([0.4, 1.0, -2.0], dtype=th.float64)
    from mowl.nn.order.manifolds import lorentz_inner
    assert lorentz_inner(x, Hyperboloid().rgrad(x, g)).abs().item() < 1e-9
    tangent = Sphere().rgrad(th.tensor([1.0, 0.0]), th.tensor([1.0, 2.0]))
    assert tangent[0].item() == pytest.approx(0.0)


def test_margin_loss_of_eq_10():
    model = DiskEmbeddings(tree_dataset(), embed_dim=3, margin=0.5, seed=0)
    pos = th.tensor([-1.0, 0.3])       # protrusions l of positives
    neg = th.relu(0.5 - th.tensor([0.1, 2.0]))     # what the module returns for negatives
    assert model.compute_loss(pos, neg).item() == pytest.approx(0.15 + (0.4 + 0.0) / 2)
    rows = th.tensor([[0, 1]])
    prot = model.module(rows, "gci0")
    assert th.allclose(model.module(rows, "gci0", neg=True), th.relu(0.5 - prot))
    with pytest.raises(NotImplementedError):
        model.module(th.tensor([[0, 1, 2]]), "gci2")


@pytest.mark.parametrize("geometry,lr", [("hyperbolic", 0.1), ("spherical", 0.1),
                                         ("euclidean", 0.1)])
def test_training_separates_ancestors_from_other_classes(geometry, lr):
    model = DiskEmbeddings(tree_dataset(), embed_dim=5, geometry=geometry,
                           learning_rate=lr, seed=0)
    assert len(model.training_datasets["gci0"]) == 15
    pos, neg = train_and_separate(model, epochs=300)
    assert pos < neg - 0.5 * abs(neg)


def test_told_axioms_get_lower_energy_and_embeddings_are_exported():
    import numpy as np
    np.random.seed(0)
    model = DiskEmbeddings(tree_dataset(), embed_dim=4, seed=0)
    rows = model.training_datasets["gci0"][:]
    before = th.relu(model.module(rows, "gci0")).mean().item()
    model.train(epochs=60)
    after = th.relu(model.module(rows, "gci0")).mean().item()
    assert after < before
    emb, _, _ = model.get_embeddings()
    assert len(emb) == len(model.class_index_dict)
    assert next(iter(emb.values())).shape == (5,)  # n + 1 coordinates on the hyperboloid


def test_ranked_by_the_joint_subsumption_benchmark():
    from tests.benchmarks.tiny import scratch_dir, tiny_dataset
    from mowl.benchmarks.comparison import compare_methods
    outcome = compare_methods(common_ranking_benchmark(), methods=["disk_embeddings"],
                              epochs=3, seeds=(0,), model_kwargs={"embed_dim": 4},
                              dataset=tiny_dataset(), workdir=scratch_dir(),
                              stop_on_error=True)
    (result,) = outcome.results
    assert 0.0 < result.metrics["all"]["f_mrr"] <= 1.0
