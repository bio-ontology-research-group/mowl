"""grounded, canonical_mixture, faithel and boxlite in the common interface."""
import itertools

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.models import CanonicalMixture  # noqa: E402
# the adapters below describe the direct implementations (the combinations of parts are their
# own geometry: tests/representation/test_recomp_solved.py)
from mowl.baselines.models.boxlite.embedding import BoxLitEEmbedding  # noqa: E402
from mowl.baselines.models.faithel.model import FaithEL  # noqa: E402
from mowl.nn.el.grounded import GroundedELModule  # noqa: E402
from mowl.representation import ConstructorUnavailable, geometry_of  # noqa: E402
from mowl.representation.library.other_regions import (  # noqa: E402
    BoxLitEGeometry, CanonicalMixtureGeometry, FaithELGeometry, GroundedGeometry,
    MaskedUnionOfBoxes)
from tests.models.test_canonical_mixture import (  # noqa: E402
    IND, NS, _write_family_abox_dataset)


# -- grounded -----------------------------------------------------------------------------
def _grounded(role_mode="translation", seed=0):
    th.manual_seed(seed)
    return GroundedELModule(nb_elements=6, nb_classes=4, nb_rels=2, embed_dim=2,
                            n_components=2, n_role_components=2, role_mode=role_mode)


def _exact_grounded():
    """A module in the exact-limit state of a small least model (product roles)."""
    from mowl.geometric.least_model import least_model
    lm = least_model(4, nb_rels=1, nb_individuals=2, gci0=[[0, 1]], gci2=[[1, 0, 2]],
                     class_assertion=[[0, 0], [1, 2]])
    module = GroundedELModule(nb_elements=6, nb_classes=4, nb_rels=1, embed_dim=2,
                              n_components=6, n_role_components=3, role_mode="product")
    module.exact_limit_init(lm.dense_labels(), lm.element_edges(), delta=0.05, seed=0)
    return module, lm


def test_grounded_geometry_shares_the_module_and_is_exact_in_the_limit():
    module, lm = _exact_grounded()
    g = geometry_of(module)
    assert isinstance(g, GroundedGeometry) and g.module is module
    labels = lm.dense_labels()
    a, b = zip(*itertools.product(range(4), repeat=2))
    cost = g.inside(g.classes(th.tensor(a)), g.classes(th.tensor(b))).reshape(4, 4)
    for x, y in itertools.product(range(4), repeat=2):
        subset = not (labels[:, x] & ~labels[:, y]).any()
        assert (cost[x, y].item() == 0.0) == subset, (x, y)
    apart = g.apart(g.classes(th.tensor(a)), g.classes(th.tensor(b))).reshape(4, 4)
    for x, y in itertools.product(range(4), repeat=2):
        assert (apart[x, y].item() == 0.0) == (not (labels[:, x] & labels[:, y]).any())
    # individuals: the module's own score, -depth, equals forward()
    data = th.tensor([[0, 0], [0, 1], [1, 2], [1, 0]])
    got = g.member(data[:, 0], g.classes(data[:, 1]))
    assert th.allclose(got[:, 0], module(data, "class_assertion"), atol=1e-6)
    inside = got[:, 0] < 0
    assert inside.tolist() == [True, True, True, False]
    # conjunction and disjunction are exact unions of boxes
    both, gap = g.conjunction(g.classes(th.tensor([0])), g.classes(th.tensor([1])))
    assert isinstance(both, MaskedUnionOfBoxes) and gap.item() == 0.0
    assert g.inside(both, g.classes(th.tensor([1]))).item() < 1e-6
    union = g.disjunction(g.classes(th.tensor([0])), g.classes(th.tensor([3])))
    assert g.inside(g.classes(th.tensor([0])), union).item() < 1e-6
    with pytest.raises(ConstructorUnavailable):
        g.negation(both)
    # product roles have no region for exists r.C
    with pytest.raises(ConstructorUnavailable):
        g.exists(th.tensor([0]), g.classes(th.tensor([1])))


def test_grounded_translation_exists_is_the_modules_and_trains_through_the_geometry():
    module = _grounded("translation")
    g = geometry_of(module)
    filler = g.classes(th.tensor([1, 2]))
    boxes = g.exists(th.tensor([0, 1]), filler)
    expected = module.exists_box(th.tensor([0, 1]), th.tensor([1, 2]))
    assert th.equal(boxes.cen, expected.cen) and th.equal(boxes.w, expected.w)
    cost = g.inside(g.classes(th.tensor([0, 0])), boxes)
    assert cost.shape == (2, 1) and (cost >= 0).all()
    before = cost.sum()
    optimiser = th.optim.Adam(module.parameters(), lr=0.05)
    for _ in range(30):
        optimiser.zero_grad()
        loss = g.inside(g.classes(th.tensor([0, 0])),
                        g.exists(th.tensor([0, 1]), g.classes(th.tensor([1, 2])))).sum()
        loss.backward()
        optimiser.step()
    assert loss.item() < before.item()
    soft = g.soft_membership(th.tensor([0, 1]), g.classes(th.tensor([0, 1])))
    assert ((soft > 0) & (soft < 1)).all()


# -- canonical mixture ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def mixture(tmp_path_factory):
    dataset = _write_family_abox_dataset(tmp_path_factory.mktemp("cm"))
    th.manual_seed(0)
    return CanonicalMixture(dataset, rank=4, lr=0.01, batch_size=2, use_cache=False)


def test_canonical_mixture_geometry_gives_the_modules_scores(mixture):
    g = geometry_of(mixture)
    assert isinstance(g, CanonicalMixtureGeometry)
    c, i = mixture.class_index_dict, mixture.individual_index_dict
    rows = th.tensor([[i[IND + "bob"], c[NS + "Father"]], [i[IND + "bob"], c[NS + "Person"]],
                      [i[IND + "bob"], c[NS + "Female"]], [i[IND + "alice"], c[NS + "Parent"]]])
    got = g.member(rows[:, 0], g.classes(rows[:, 1]))
    expected = mixture.module.class_assertion_score(rows)
    finite = th.isfinite(expected)
    assert finite.any()
    assert th.allclose(got[finite, 0], expected[finite], atol=1e-5)
    assert (th.isinf(got[:, 0]) == th.isinf(expected)).all()
    # sets of canonical elements: inclusion is exact and equals the canonical model
    cm = mixture.structure
    names = [n for n, _ in sorted(c.items(), key=lambda kv: kv[1])]
    ids = th.arange(len(names))
    a, b = zip(*itertools.product(range(len(names)), repeat=2))
    cost = g.inside(g.classes(ids[list(a)]), g.classes(ids[list(b)]))
    assert set(cost.unique().tolist()) <= {0.0, 1.0}
    father, parent, person = (c[NS + n] for n in ("Father", "Parent", "Person"))
    assert g.inside(g.classes(th.tensor([father])), g.classes(th.tensor([parent]))).item() \
        == 0.0
    assert g.inside(g.classes(th.tensor([person])), g.classes(th.tensor([father]))).item() \
        == 1.0
    assert cm.M.shape[1] == len(names)
    # existential restriction and a role assertion: the module's own -log probability
    has_child = mixture.object_property_index_dict[NS + "hasChild"]
    pairs = th.tensor([[i[IND + "bob"], has_child, i[IND + "carl"]],
                       [i[IND + "carl"], has_child, i[IND + "diana"]]])
    target = g.exists(pairs[:, 1], g.individuals(pairs[:, 2]))
    got = g.member(pairs[:, 0], target)
    expected = mixture.module.object_property_assertion_score(pairs)
    finite = th.isfinite(expected)
    assert th.allclose(got[finite, 0], expected[finite], atol=1e-5)
    # conjunction / negation of sets are exact
    both, gap = g.conjunction(g.classes(th.tensor([father])), g.classes(th.tensor([person])))
    assert g.inside(both, g.classes(th.tensor([father]))).item() == 0.0 and gap.item() == 0.0
    assert g.apart(g.classes(th.tensor([father])), g.negation(
        g.classes(th.tensor([father])))).item() == 0.0


def test_canonical_mixture_parameters_are_shared(mixture):
    g = geometry_of(mixture)
    mixture.module.zero_grad()
    rows = th.tensor([[0, 0]])
    g.member(rows[:, 0], g.classes(rows[:, 1])).sum().backward()
    assert mixture.module.V.grad is not None or mixture.module.U.grad is not None


# -- faithel -------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def faithel(tmp_path_factory):
    dataset = _write_family_abox_dataset(tmp_path_factory.mktemp("fa"))
    return FaithEL(dataset, batch_size=8, learning_rate=0.01, neg_sampling=False, seed=0)


def test_faithel_tbox_costs_and_abox_scores_are_the_models(faithel):
    # faithel.module is a plain torch Module: geometry_of finds the adapter by the model's
    # own class, and gives the same geometry as building it from the model.
    g = geometry_of(faithel)
    assert isinstance(g, FaithELGeometry)
    assert isinstance(FaithELGeometry(faithel), FaithELGeometry)
    n = len(faithel.class_index_dict)
    pairs = th.tensor(list(itertools.product(range(n), repeat=2)))
    got = g.inside(g.classes(pairs[:, 0]), g.classes(pairs[:, 1]))
    assert th.equal(got[:, 0], faithel.axiom_costs("gci0", pairs))
    triples = th.tensor([[a, 0, b] for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]])
    exists = g.exists(triples[:, 1], g.classes(triples[:, 2]))
    assert th.equal(g.inside(g.classes(triples[:, 0]), exists)[:, 0],
                    faithel.axiom_costs("gci2", triples))
    # individuals: the learned vectors
    ind = faithel.individual_index_dict
    rows = th.tensor([[ind[IND + "bob"], faithel.class_index_dict[NS + "Father"]],
                      [ind[IND + "alice"], faithel.class_index_dict[NS + "Female"]]])
    got = g.member(rows[:, 0], g.classes(rows[:, 1]))
    assert th.allclose(got[:, 0], faithel.axiom_costs("class_assertion", rows))
    role = th.tensor([[ind[IND + "bob"], 0, ind[IND + "carl"]]])
    got = g.member(role[:, 0], g.exists(role[:, 1], g.individuals(role[:, 2])))
    assert th.allclose(got[:, 0], faithel.axiom_costs("object_property_assertion", role))
    # the parameters are the model's
    faithel.module.zero_grad()
    g.member(rows[:, 0], g.classes(rows[:, 1])).sum().backward()
    assert faithel.module.individuals.weight.grad is not None


# -- boxlite -------------------------------------------------------------------------------
def _embedding(seed=0, margin=0.01):
    rng = np.random.default_rng(seed)
    nc, nr, ni, d = 3, 2, 4, 5

    def boxes(n):
        lower = rng.uniform(-0.8, 0.0, (n, d))
        return lower, lower + rng.uniform(0.2, 1.0, (n, d))
    cl, cu = boxes(nc)
    hl, hu = boxes(nr)
    tl, tu = boxes(nr)
    bl, bu = boxes(nr)
    return BoxLitEEmbedding(pos=rng.uniform(-0.5, 0.5, (ni, d)),
                            bump=rng.uniform(-0.3, 0.3, (ni, d)), concept_lower=cl,
                            concept_upper=cu, head_lower=hl, head_upper=hu, tail_lower=tl,
                            tail_upper=tu, bump_lower=bl, bump_upper=bu, s_omega=1.0,
                            margin=margin)


def test_boxlite_geometry_gives_the_papers_losses():
    embedding = _embedding()
    g = geometry_of(embedding)
    assert isinstance(g, BoxLitEGeometry)
    nbasic = 3 + 2 * 2
    assert g.num_classes == nbasic and g.num_individuals == 4
    for basic in range(nbasic):
        ids = th.arange(4)
        got = g.member(ids, g.classes(th.full((4,), basic)))
        expected = embedding.concept_loss(basic)
        assert np.allclose(got[:, 0].numpy(), expected, atol=1e-5), basic
    r = th.tensor([1, 0])
    a, b = th.tensor([0, 2]), th.tensor([3, 1])
    got = g.member(a, g.exists(r, g.individuals(b)))
    expected = np.array([embedding.role_loss(1, [0], [3])[0],
                         embedding.role_loss(0, [2], [1])[0]])
    assert np.allclose(got[:, 0].numpy(), expected, atol=1e-5)
    # existential restrictions of DL-Lite are the boxes of the basic concepts
    some = g.exists(th.tensor([0]), g.universe())
    assert th.allclose(some.centre, g.classes(th.tensor([3])).centre)
    inverse = g.exists(th.tensor([1]), g.universe(), inverse=True)
    assert th.allclose(inverse.centre, g.classes(th.tensor([3 + 2 + 1])).centre)
    with pytest.raises(ConstructorUnavailable):
        g.exists(th.tensor([0]), g.classes(th.tensor([0])))


def test_boxlite_inclusion_conjunction_and_negation():
    g = geometry_of(_embedding(seed=1, margin=0.0))
    c = g.classes(th.tensor([0, 1]))
    inside = g.inside(c, c)
    assert th.allclose(inside, th.zeros_like(inside))
    both, gap = g.conjunction(g.classes(th.tensor([0])), g.classes(th.tensor([1])))
    assert th.allclose(g.inside(both, g.classes(th.tensor([0]))), th.zeros(1, 1), atol=1e-6)
    # the complement box is [-s - L, s - U]
    not_c = g.negation(g.classes(th.tensor([0])))
    emb = g.embedding
    assert np.allclose(not_c.lower[0].numpy(), -1.0 - emb.concept_lower[0], atol=1e-6)
    assert np.allclose(not_c.upper[0].numpy(), 1.0 - emb.concept_upper[0], atol=1e-6)
    assert th.is_tensor(gap)
    with pytest.raises(ConstructorUnavailable):
        g.disjunction(c, c)
