"""CatE (Zhapa-Camacho & Hoehndorf, arXiv:2305.07163v3): order-embedding module, the
zero-loss guarantee (Theorem 2), training and scoring end to end, unseen concepts, the
T3 Helly pattern and the benchmark runner."""

import os

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

import mowl.datasets  # noqa: E402,F401  (import before mowl.nn, see COMMON.md)
from mowl.datasets import Dataset, PathDataset  # noqa: E402
from mowl.owlapi import OWLAPIAdapter  # noqa: E402
from mowl.models.cate.model import CatE  # noqa: E402
from mowl.nn.graph.cate import CatEModule, order_margin_loss  # noqa: E402
from mowl.projection.categorical.lattice import CatELattice, UnseenConceptError  # noqa: E402

NS = "http://mowl.borg/cate-model#"
ad = OWLAPIAdapter()
df = ad.data_factory
SUB = ad.create_subclass_of
NOT = ad.create_complement_of
OR = ad.create_object_union_of
AND = ad.create_object_intersection_of
SOME = ad.create_object_some_values_from
ALL = ad.create_object_all_values_from
T3_PATH = os.path.join(os.path.dirname(__file__), "..", "geometric", "fixtures", "t3.owl")
GEO = "http://mowl.borg/geo#"


def c(name):
    return ad.create_class(NS + name)


def i(name):
    return df.getOWLNamedIndividual(ad.create_class(NS + name).getIRI())


EATS = ad.create_object_property(NS + "eats")
OWNS = ad.create_object_property(NS + "owns")
_COUNTER = [0]


def ontology(axioms):
    _COUNTER[0] += 1
    ont = ad.create_ontology(f"{NS}o{_COUNTER[0]}")
    for ax in axioms:
        ad.owl_manager.addAxiom(ont, ax)
    return ont


def pets_axioms():
    """A tiny ALC ontology with negation, disjunction and both restrictions."""
    return [SUB(c("Pet"), OR(c("Cat"), c("Dog"))),
            SUB(c("Cat"), c("Animal")), SUB(c("Dog"), c("Animal")),
            SUB(c("Cat"), NOT(c("Dog"))),
            SUB(c("Cat"), SOME(EATS, c("Fish"))),
            SUB(c("Dog"), ALL(EATS, c("Meat"))),
            SUB(c("Fish"), NOT(c("Meat"))),
            df.getOWLClassAssertionAxiom(c("Cat"), i("tom")),
            df.getOWLClassAssertionAxiom(c("Dog"), i("rex")),
            df.getOWLObjectPropertyAssertionAxiom(OWNS, i("alice"), i("tom"))]


def pets_dataset(validation=True):
    valid = ontology([SUB(c("Pet"), c("Animal"))]) if validation else None
    return Dataset(ontology(pets_axioms()), validation=valid)


def scratch_file(name):
    base = os.environ.get("TMPDIR") or os.path.expanduser("~/tmp")
    d = os.path.join(base, "mowl_cate_tests")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


# ---------------------------------------------------------------------------------------
# Module and loss semantics
# ---------------------------------------------------------------------------------------

def test_order_violation_is_zero_iff_coordinatewise_below():
    m = CatEModule(3, 2)
    with th.no_grad():
        m.node_embedding.weight.copy_(th.tensor([[0., 0.], [1., 1.], [2., -1.]]))
    s = m.order_violation(th.tensor([0, 1, 0, 2]), th.tensor([1, 0, 2, 1]))
    assert s[0].item() == 0.0                       # (0,0) ⪯ (1,1)
    assert s[1].item() == pytest.approx(np.sqrt(2))  # (1,1) vs (0,0): ||(1,1)||
    assert s[2].item() == pytest.approx(1.0)        # (0,0) vs (2,-1): only y violated
    assert s[3].item() == pytest.approx(1.0)        # (2,-1) vs (1,1)
    rows = th.tensor([[0, 1], [1, 0]])
    assert th.equal(m(rows, "gci0"), m(rows, "class_assertion"))
    with pytest.raises(NotImplementedError):
        m(th.tensor([[0, 1, 2]]), "gci1")


def test_existential_lookup_and_missing_existential():
    m = CatEModule(5, 2)
    m.set_existentials([0, 1], [2, 2], [3, 4])
    assert m.existential_ids(th.tensor([0, 1, 0]), th.tensor([2, 2, 1])).tolist() == [3, 4, -1]
    m(th.tensor([[0, 1, 2]]), "gci2")
    with pytest.raises(UnseenConceptError):
        m(th.tensor([[0, 0, 1]]), "gci2")


def test_margin_loss_aggregations_and_mask():
    pos = th.tensor([0.5, 0.0])
    neg = th.tensor([[0.0, 2.0], [0.4, 0.4]])
    # Eq. 7: one hinge per negative.
    expected = pos.mean() + th.relu(1.0 - neg).mean()
    assert order_margin_loss(pos, neg, 1.0).item() == pytest.approx(expected.item())
    # Reference code: hinge of the mean negative score.
    ref = pos.mean() + th.relu(1.0 - neg.mean(dim=1)).mean()
    assert order_margin_loss(pos, neg, 1.0, "mean").item() == pytest.approx(ref.item())
    mask = th.tensor([[False, True], [True, True]])
    masked = pos.mean() + (0.0 + 0.6 + 0.6) / 3
    assert order_margin_loss(pos, neg, 1.0, neg_mask=mask).item() == pytest.approx(masked)


# ---------------------------------------------------------------------------------------
# Theorem 2: zero loss => every morphism is order-preserved
# ---------------------------------------------------------------------------------------

def _downset_embedding(lattice):
    """f(x)_z = 1 iff z ⊑ x (reflexive-transitive closure of the morphisms)."""
    n = len(lattice)
    succ = [[] for _ in range(n)]
    for s, d in lattice.edges.tolist():
        succ[s].append(d)
    f = np.zeros((n, n), dtype=np.float32)
    for z in range(n):
        stack, seen = [z], {z}
        while stack:
            x = stack.pop()
            for y in succ[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        for x in seen:
            f[x, z] = 1.0
    return f


def _full_loss(module, lattice, margin):
    """Eq. 7 summed over every morphism and every non-morphism A ⊑ B' (B' ≠ A)."""
    n = len(lattice)
    pos = lattice.edges
    adj = th.zeros(n, n, dtype=th.bool)
    adj[pos[:, 0], pos[:, 1]] = True
    adj |= th.eye(n, dtype=th.bool)
    neg = (~adj).nonzero()
    with th.no_grad():
        positive = module.order_violation(pos[:, 0], pos[:, 1]).sum()
        negative = th.relu(margin - module.order_violation(neg[:, 0], neg[:, 1])).sum()
        return (positive + negative).item()


def test_zero_loss_embedding_preserves_every_morphism():
    ds = PathDataset(T3_PATH)
    lat = CatELattice(ds.ontology, classes=ds.classes.as_str, saturation_steps=None,
                      transitive_closure="full")
    f = _downset_embedding(lat)
    module = CatEModule(len(lat), len(lat))
    with th.no_grad():
        module.node_embedding.weight.copy_(th.from_numpy(f))
    assert _full_loss(module, lat, margin=1.0) == 0.0
    # Independent check of the conclusion: f(A) <= f(B) coordinatewise for A ⊑ B.
    for s, d in lat.edges.tolist():
        assert np.all(f[s] <= f[d]), (lat.names[s], lat.names[d])


def test_zero_loss_needs_a_transitively_closed_lattice():
    # Without the closure, A ⊑ B ⊑ C makes A ⊑ C a non-morphism with s(A, C) = 0, so any
    # embedding that preserves the morphisms has L >= γ.
    lat = CatELattice(ontology([SUB(c("A"), c("B")), SUB(c("B"), c("C"))]))
    a, cc = lat.class_to_id[NS + "A"], lat.class_to_id[NS + "C"]
    assert not lat.has_edge(a, cc)
    f = _downset_embedding(lat)
    module = CatEModule(len(lat), len(lat))
    with th.no_grad():
        module.node_embedding.weight.copy_(th.from_numpy(f))
    assert module.order_violation(lat.edges[:, 0], lat.edges[:, 1]).max().item() == 0.0
    assert _full_loss(module, lat, margin=1.0) >= 1.0


# ---------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained():
    # With the closure (Tr) the morphisms do not compete with non-morphism composites
    # such as {tom} ⊑ Animal (see test_zero_loss_needs_a_transitively_closed_lattice);
    # without it, Cat ⊑ Animal keeps a violation of about 0.5 on this ontology.
    th.manual_seed(0)
    model = CatE(pets_dataset(), embed_dim=16, margin=0.5, num_negs=4, learning_rate=0.01,
                 batch_size=128, saturation_steps=1, transitive_closure=True,
                 existentials={NS + "eats": [NS + "Fish", NS + "Meat", NS + "Cat"]},
                 model_filepath=scratch_file("pets.pt"))
    history = model.train(epochs=300, validate_every=50)
    return model, history


def test_training_decreases_loss_and_saves(trained):
    model, history = trained
    assert history[-1] < history[0]
    assert os.path.exists(model.model_filepath)
    assert model.validation_history and model.validation_queries()
    assert all(0 < mrr <= 1 for _, mrr in model.validation_history)
    w = model.module.node_embedding.weight
    assert w.shape == (len(model.lattice), 16) and th.isfinite(w).all()


def test_scores_of_axioms(trained):
    model, _ = trained
    true = [model.score(ax) for ax in pets_axioms()]
    false = [model.score(SUB(c(x), c(y))) for x, y in
             (("Cat", "Dog"), ("Dog", "Cat"), ("Fish", "Cat"), ("Animal", "Fish"))]
    assert max(true) < 0.25, true
    assert np.mean(true) < np.mean(false), (true, false)
    lat = model.lattice
    cat, dog = lat.class_to_id[NS + "Cat"], lat.class_to_id[NS + "Dog"]
    tom = lat.individual_to_id[NS + "tom"]
    assert model.score_membership([tom], [cat]).item() < \
        model.score_membership([tom], [dog]).item()
    # C ⊑ ∃r.D through the element ∃eats.D, in el_scoring row layout.
    eats = model.object_property_index_dict[NS + "eats"]
    fish = lat.class_to_id[NS + "Fish"]
    s = model.score_existential([cat], [eats], [fish])
    assert s.item() == pytest.approx(model.score(SUB(c("Cat"), SOME(EATS, c("Fish")))))
    rows = th.tensor([[cat, eats, fish]])
    assert model.module(rows, "gci2").item() == pytest.approx(s.item())


def test_unseen_complex_concept_raises(trained):
    model, _ = trained
    with pytest.raises(UnseenConceptError, match="not an element"):
        model.score(SUB(c("Cat"), AND(c("Fish"), c("Meat"))))
    with pytest.raises(UnseenConceptError):
        model.score(SUB(c("Cat"), SOME(OWNS, c("Fish"))))
    lat = model.lattice
    owns = model.object_property_index_dict[NS + "owns"]
    with pytest.raises(UnseenConceptError):
        model.score_existential([lat.class_to_id[NS + "Cat"]], [owns],
                                [lat.class_to_id[NS + "Fish"]])
    with pytest.raises(TypeError):
        model.score(df.getOWLDeclarationAxiom(c("Cat")))


def test_obvious_morphisms_filter():
    """obvious_morphisms=False drops the trivial morphisms of complex elements, as
    remove_obvious_morphisms.py of the reference code does to produce the released
    training graphs; named classes keep them."""
    lat = CatELattice(pets_dataset().ontology, obvious_morphisms=False)
    bot, top = lat.bot_id, lat.top_id
    cat = lat.class_to_id[NS + "Cat"]
    complex_ids = [i for i, n in enumerate(lat.names)
                   if n.startswith(("ObjectIntersectionOf", "ObjectUnionOf",
                                    "ObjectComplementOf"))]
    edges = set(map(tuple, lat.edges.tolist()))
    for i in complex_ids:
        assert (bot, i) not in edges
        assert (i, top) not in edges
    assert (bot, cat) in edges and (cat, top) in edges
    # The default keeps them (the paper: every element has bot ⊑ X ⊑ top).
    lat_default = CatELattice(pets_dataset().ontology)
    edges = set(map(tuple, lat_default.edges.tolist()))
    assert all((bot, i) in edges and (i, top) in edges for i in complex_ids)


def test_punned_individual_shares_the_class_element():
    """An individual whose IRI is also a class (punning, as throughout the ORE1 data)
    is the class's element, as in the reference code whose node names are strings."""
    famous = c("FamousHuman")
    axioms = [SUB(c("Person"), famous),
              df.getOWLClassAssertionAxiom(famous, i("clinton")),
              df.getOWLClassAssertionAxiom(c("Person"), i("FamousHuman"))]
    ds = Dataset(ontology(axioms))
    model = CatE(ds, existentials=None, model_filepath=scratch_file("punned.pt"))
    lat = model.lattice
    assert lat.individual_to_id[NS + "FamousHuman"] == lat.class_to_id[NS + "FamousHuman"]
    assert lat.individual_to_id[NS + "clinton"] != lat.class_to_id.get(NS + "Person")
    assert len(model.validation_queries()) == 0  # no validation ontology
    s = model.score(df.getOWLClassAssertionAxiom(c("Person"), i("clinton")))
    assert s >= 0.0


def test_embedding_dictionaries(trained):
    model, _ = trained
    assert NS + "Cat" in model.class_embeddings
    assert NS + "tom" in model.individual_embeddings
    assert len(model.node_embeddings) == len(model.lattice)


def test_reference_training_variant_runs():
    model = CatE(pets_dataset(validation=False), embed_dim=8, batch_size=64,
                 learning_rate=0.01, negative_aggregation="mean", filter_negatives=False,
                 cyclic_lr=False, complete_negation=False,
                 model_filepath=scratch_file("pets_ref.pt"))
    history = model.train(epochs=5)
    assert len(history) == 5 and all(np.isfinite(history))
    with pytest.raises(ValueError):
        CatE(pets_dataset(), negative_aggregation="sum")


# ---------------------------------------------------------------------------------------
# T3 Helly pattern
# ---------------------------------------------------------------------------------------

@pytest.mark.slow
def test_t3_trained_order_respects_disjointness():
    """A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3, A ⊓ B ⊑ ⊥ (etc.): with S1 and the full
    transitive closure every morphism (including A ⊑ ¬B and A ⊓ B ⊑ ⊥) is satisfied up to
    a small violation, and A, B, C are not ordered below each other. The coordinatewise
    meet of f(A) and f(B) is not required to lie below f(⊥)."""
    th.manual_seed(0)
    model = CatE(PathDataset(T3_PATH), embed_dim=16, margin=0.5, learning_rate=0.01,
                 batch_size=256, saturation_steps=None, transitive_closure="full",
                 model_filepath=scratch_file("t3.pt"))
    model.train(epochs=1000)
    lat = model.lattice
    s = model.score_edges(lat.edges[:, 0], lat.edges[:, 1])
    assert s.max().item() < 0.1, s.max().item()
    g = {n: ad.create_class(GEO + n) for n in ("A", "B", "C")}
    for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
        assert model.score(SUB(g[x], NOT(g[y]))) < 0.05
        assert model.score(SUB(AND(g[x], g[y]), ad.create_class(
            "http://www.w3.org/2002/07/owl#Nothing"))) < 0.1
        assert model.score(SUB(g[x], g[y])) > 0.25
        assert model.score(SUB(g[y], g[x])) > 0.25


# ---------------------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------------------

def test_benchmark_runner_family_and_tiny_runs():
    from mowl.benchmarks.runner import family_for, run_benchmark
    from tests.benchmarks.tiny import NS as TNS, tiny_dataset
    assert family_for(CatE).name == "cate"
    ds = tiny_dataset()
    kwargs = {"embed_dim": 8, "batch_size": 64, "learning_rate": 0.01}
    # The completion benchmark ranks the released query lists, which the tiny dataset
    # does not have; queries=None falls back to the test axioms.
    res = run_benchmark("cate-go-completion", method="cate", dataset=ds,
                        model_kwargs=kwargs, epochs=4,
                        protocol_kwargs={"queries": None, "closure": False})
    assert set(res.metrics["all"]) == {"f_mr", "f_hits@10", "f_hits@100", "f_auc"}
    assert res.protocol_info["queries"] > 0
    res = run_benchmark("cate-yeast-ppi", method="cate", dataset=ds, epochs=4,
                        model_kwargs={**kwargs,
                                      "existentials": {TNS + "r": "evaluation_classes"}},
                        protocol_kwargs={"relation": TNS + "r"})
    assert res.protocol_info["queries"] == 1
    assert 0 < res.metrics["all"]["f_mrr"] <= 1
