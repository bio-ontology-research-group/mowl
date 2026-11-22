"""Query embedding models on ontologies: Query2Box, BetaE, ConE, AConE."""
import numpy as np
import pytest
import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.evaluation import LogicalQueryAnswering, sample_queries
from mowl.models import AConE, BetaE, ConE, Query2Box
from mowl.models.query.graph import ISA, ISA_INV, role_relation
from mowl.nn.query.geometry import UnsupportedOperator
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl/qm#"
ad = OWLAPIAdapter()
ALL = (Query2Box, BetaE, ConE, AConE)
_COUNT = [0]


def _iri(tag):
    _COUNT[0] += 1
    return f"http://mowl/qm/{tag}{_COUNT[0]}"


def _dataset():
    """Animal ⊒ Mammal ⊒ {Dog, Cat}, Animal ⊒ Bird; Dog ⊑ ∃eats.Meat, Cat ⊑ ∃eats.Meat,
    Bird ⊑ ∃eats.Seed; Pet ≡ Dog ⊔ Cat is skipped; rex: Dog, tom: Cat, eats(rex, bone)."""
    ont = ad.create_ontology(_iri("o"))
    c = {n: ad.create_class(NS + n) for n in
         "Animal Mammal Dog Cat Bird Meat Seed Pet Food".split()}
    eats = ad.create_object_property(NS + "eats")
    ax = HashSet()
    for sub, sup in [("Mammal", "Animal"), ("Bird", "Animal"), ("Dog", "Mammal"),
                     ("Cat", "Mammal"), ("Meat", "Food"), ("Seed", "Food")]:
        ax.add(ad.create_subclass_of(c[sub], c[sup]))
    for sub, f in [("Dog", "Meat"), ("Cat", "Meat"), ("Bird", "Seed")]:
        ax.add(ad.create_subclass_of(c[sub], ad.create_object_some_values_from(eats, c[f])))
    ax.add(ad.create_subclass_of(c["Pet"], ad.create_object_union_of(c["Dog"], c["Cat"])))
    rex, tom, bone = (ad.create_individual(NS + n) for n in ("rex", "tom", "bone"))
    ax.add(ad.create_class_assertion(c["Dog"], rex))
    ax.add(ad.create_class_assertion(c["Cat"], tom))
    ax.add(ad.create_object_property_assertion(eats, rex, bone))
    ad.owl_manager.addAxioms(ont, ax)
    return Dataset(ont), c, eats


def test_graph_of_the_ontology():
    ds, c, eats = _dataset()
    m = Query2Box(ds, embed_dim=8)
    cid, n = m._class_ids, m.num_classes
    t = set(map(tuple, m.triples.tolist()))
    e = lambda name: cid[NS + name]                                      # noqa: E731
    assert (e("Dog"), ISA, e("Mammal")) in t and (e("Mammal"), ISA_INV, e("Dog")) in t
    assert (e("Dog"), ISA, e("Animal")) in t                           # transitive closure
    assert (e("Animal"), ISA, e("Animal")) in t                        # reflexive on classes
    r = role_relation(m._role_ids[NS + "eats"])
    assert (e("Dog"), r, e("Meat")) in t and (e("Meat"), r ^ 1, e("Dog")) in t
    rex = n + m._individual_ids[NS + "rex"]
    assert (rex, ISA, e("Dog")) in t and (rex, ISA, e("Animal")) in t
    assert (rex, r, n + m._individual_ids[NS + "bone"]) in t
    assert m.skipped_axioms == 1                                       # Pet ⊑ Dog ⊔ Cat
    # no hierarchy closure when asked
    plain = Query2Box(ds, embed_dim=8, hierarchy_closure=False)
    assert (e("Dog"), ISA, e("Animal")) not in set(map(tuple, plain.triples.tolist()))


def _mean_rank_of_answers(model, c):
    """Mean rank (0 = best) of the true answers of 'everything below Mammal'."""
    scores = model.answer(c["Mammal"], top_k=len(model.entity_names))
    order = [iri for iri, _ in scores]
    true = [NS + n for n in ("Mammal", "Dog", "Cat", "rex", "tom")]
    return float(np.mean([order.index(t) for t in true]))


@pytest.mark.parametrize("cls", ALL)
def test_training_improves_the_ranking_of_answers(cls):
    th.manual_seed(0)
    ds, c, eats = _dataset()
    # ConE collapses with learning rates above about 5e-3 at this size
    lr = 3e-3 if cls is ConE else 1e-2
    m = cls(ds, embed_dim=16, learning_rate=lr, batch_size=30, negatives=8, seed=0,
            structures=("1p",))
    before = _mean_rank_of_answers(m, c)
    m.train(epochs=60 if cls is not ConE else 100, steps_per_epoch=5)
    assert np.mean(m.loss_history[-3:]) < 0.5 * np.mean(m.loss_history[:3])
    after = _mean_rank_of_answers(m, c)
    assert after <= 3.0 and after < before, (before, after)      # 2.0 is a perfect ranking


@pytest.mark.parametrize("cls", ALL)
def test_all_structures_train_and_lower_the_loss(cls):
    th.manual_seed(0)
    ds, c, eats = _dataset()
    m = cls(ds, embed_dim=8, learning_rate=3e-3, batch_size=30, negatives=4, seed=0)
    assert {"1p", "2p", "3p", "2i", "3i", "ip", "pi"} <= set(m.training_structures)
    m.train(epochs=20, steps_per_epoch=2)
    assert np.mean(m.loss_history[-3:]) < np.mean(m.loss_history[:3])


def test_query2box_ranks_the_subclasses_first_after_training():
    th.manual_seed(0)
    ds, c, eats = _dataset()
    m = Query2Box(ds, embed_dim=16, learning_rate=5e-3, batch_size=40, negatives=8, seed=0)
    m.train(epochs=30, steps_per_epoch=5)
    scores = dict(m.answer(c["Mammal"], top_k=len(m.entity_names)))
    assert scores[NS + "Dog"] > scores[NS + "Seed"]
    row = lambda a, b: th.tensor([[m._class_ids[NS + a], m._class_ids[NS + b]]])  # noqa: E731
    assert float(m.axiom_costs("gci0", row("Dog", "Mammal"))) < \
        float(m.axiom_costs("gci0", row("Seed", "Mammal")))


def test_axiom_forms_and_operators():
    ds, c, eats = _dataset()
    m = Query2Box(ds, embed_dim=8)
    cid = m._class_ids
    r = m._role_ids[NS + "eats"]
    gci2 = th.tensor([[cid[NS + "Dog"], r, cid[NS + "Meat"]]])
    assert m.axiom_costs("gci2", gci2).shape == (1,)
    ca = th.tensor([[m._individual_ids[NS + "rex"], cid[NS + "Dog"]]])
    assert m.axiom_costs("class_assertion", ca).shape == (1,)
    from mowl.evaluation.model_scoring import UnsupportedAxiomForm
    with pytest.raises(UnsupportedAxiomForm):
        m.axiom_costs("gci1", th.zeros((1, 3), dtype=th.long))
    # Query2Box has no ¬ and no ∀ (¬∃¬); the others answer them
    neg = ad.create_complement_of(c["Dog"])
    with pytest.raises(UnsupportedOperator):
        m.answer(neg)
    with pytest.raises(UnsupportedOperator):
        m.answer(ad.create_object_all_values_from(eats, c["Meat"]))
    for cls in (BetaE, ConE, AConE):
        other = cls(ds, embed_dim=8)
        assert len(other.answer(neg, top_k=3)) == 3
        assert len(other.answer(ad.create_object_all_values_from(eats, c["Meat"]), 3)) == 3
        assert "2in" in other.training_structures
    assert "2in" not in m.training_structures
    # union, intersection and existential restriction of class expressions all run
    q = ad.create_object_union_of(
        c["Bird"], ad.create_object_intersection_of(
            c["Mammal"], ad.create_object_some_values_from(eats, c["Meat"])))
    assert len(m.answer(q, top_k=5)) == 5


def test_ontology_without_edges_cannot_be_trained():
    ont = ad.create_ontology(_iri("empty"))
    ad.owl_manager.addAxiom(ont, ad.create_subclass_of(
        ad.create_object_some_values_from(ad.create_object_property(NS + "r"),
                                          ad.create_class(NS + "X")), ad.create_class(NS + "Y")))
    m = Query2Box(Dataset(ont), embed_dim=4, hierarchy_closure=False)
    with pytest.raises(ValueError):
        m.train(epochs=1)


def test_score_query_feeds_the_shared_query_answering_regime():
    th.manual_seed(0)
    ds, c, eats = _dataset()
    m = ConE(ds, embed_dim=8, batch_size=20, negatives=4)
    m.train(epochs=3, steps_per_epoch=3)
    known = m.triples
    complete = np.concatenate([known, [[m._class_ids[NS + "Cat"], role_relation(0),
                                        m._class_ids[NS + "Seed"]]]])
    queries = sample_queries(known, complete, structures=("1p",), num_queries=3, seed=0)
    regime = LogicalQueryAnswering(queries)
    result = regime.evaluate(m)
    assert 0 < result["average"]["mrr"] <= 1
    # a negation structure in the nested format is answered too
    structure = (("e", ("r",)), ("e", ("r", "n")))
    scores = m.score_query(structure, [((0, (ISA_INV,)), (1, (ISA_INV, -2)))])
    assert scores.shape == (1, m.graph.num_entities)
    with pytest.raises(UnsupportedOperator):
        Query2Box(ds, embed_dim=4).score_query(structure, [((0, (1,)), (1, (1, -2)))])


@pytest.mark.parametrize("cls", ALL)
def test_one_regime_ranks_the_query_models(cls):
    """The joint common-* ranking benchmarks run on every query model."""
    from mowl.benchmarks.comparison import compare_methods
    from tests.benchmarks.test_common import _bench
    from tests.benchmarks.tiny import scratch_dir, tiny_dataset
    key = {Query2Box: "query2box", BetaE: "betae", ConE: "cone_qe", AConE: "acone"}[cls]
    for form in ("gci0", "gci2"):
        outcome = compare_methods(_bench(form), methods=[key], epochs=2, seeds=(0,),
                                  model_kwargs={"embed_dim": 8}, dataset=tiny_dataset(),
                                  workdir=scratch_dir(), stop_on_error=True)
        (result,) = outcome.results
        assert 0.0 < result.metrics["all"]["f_mrr"] <= 1.0
        assert result.protocol_info["scoring"] == "model"


def test_sampled_training_queries_have_their_answer_among_the_exact_answers():
    """Every training structure (multi-hop and negation included) is sampled, and the
    positive is an answer of the sampled query on the graph."""
    from mowl.models.query import sampling
    from mowl.nn.query.expression import answers
    ds, c, eats = _dataset()
    m = BetaE(ds, embed_dim=4)
    rng = np.random.default_rng(0)
    assert set(m.training_structures) == set(sampling.structures(True))
    for name, template in m.training_structures.items():
        for _ in range(20):
            sample = sampling.sample_query(template, m.graph, rng, m._pool)
            assert sample is not None, name
            query, found, answer = sample
            assert answer in answers(query, m.graph, m.graph.universe), name
            assert found == answers(query, m.graph, m.graph.universe), name


def test_acone_fits_multi_step_queries_when_trained_long_enough():
    """1p, 2p and 3p isa queries: with enough step size the ranking is perfect (the mean
    rank of the 5 answers is 2.0); with 3e-3 and the same steps it is not yet."""
    th.manual_seed(0)
    ds, c, eats = _dataset()
    m = AConE(ds, embed_dim=16, learning_rate=6e-3, batch_size=30, negatives=8, seed=0,
              structures=("1p", "2p", "3p"))
    m.train(epochs=100, steps_per_epoch=5)
    assert _mean_rank_of_answers(m, c) <= 3.0


def test_collapse_of_conE_is_reported():
    """At learning rate 1e-2 the tanh of the ConE projection saturates, every query gets
    the same cone and the loss is log 2; the model says so."""
    th.manual_seed(0)
    ds, c, eats = _dataset()
    m = ConE(ds, embed_dim=16, learning_rate=1e-2, batch_size=30, negatives=8, seed=0,
             structures=("1p",))
    with pytest.warns(UserWarning, match="collapsed"):
        m.train(epochs=60, steps_per_epoch=5)
    assert abs(m.loss_history[-1] - np.log(2)) < 0.02
    scores = m.score_query(("e", ("r",)), [(0, (1,)), (3, (1,))])
    assert th.allclose(scores[0], scores[1], atol=1e-3)         # the same cone for all queries


def test_geometry_options_reach_the_geometry_from_the_model():
    ds = _dataset()
    m = AConE(ds, embed_dim=8, signed_negation=True, aperture="clip")
    assert m.module.signed_negation is True and m.module.aperture_mode == "clip"
    m = AConE(ds, embed_dim=8)
    assert m.module.signed_negation is False and m.module.aperture_mode == "wrap"
    assert Query2Box(ds, embed_dim=8, clamp_extent=True).module.clamp_extent is True
    assert Query2Box(ds, embed_dim=8).module.clamp_extent is False
    assert ConE(ds, embed_dim=8, signed_negation=True).module.signed_negation is True
