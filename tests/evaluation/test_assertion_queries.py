"""The FaithEL family dataset and the hits@k regime of concept and role assertion queries."""

import pathlib

import pytest
import torch as th

from mowl.benchmarks import get_benchmark
from mowl.catalog import get_regime
from mowl.datasets.builtin.faithel_family import FaithELFamilyDataset
from mowl.evaluation.assertion_queries import AssertionQueryHitsRegime

NS = "http://example.org/f#"


def _raw(tmp_path):
    """Person <- Parent <- Mother; Mother(m0..m4); hasChild(m_i, c_i); a lonely individual."""
    import java.io
    from java.util import HashSet
    from org.semanticweb.owlapi.formats import FunctionalSyntaxDocumentFormat
    from org.semanticweb.owlapi.model import IRI
    from mowl.owlapi import OWLAPIAdapter
    ad = OWLAPIAdapter()
    ont = ad.create_ontology(NS + "raw")
    person, parent, mother = (ad.create_class(NS + n) for n in ("Person", "Parent", "Mother"))
    has_child = ad.create_object_property(NS + "hasChild")
    axs = HashSet()
    axs.add(ad.create_subclass_of(parent, person))
    axs.add(ad.create_subclass_of(mother, parent))
    for i in range(5):
        axs.add(ad.create_class_assertion(mother, ad.create_individual(f"{NS}m{i}")))
        axs.add(ad.create_object_property_assertion(
            has_child, ad.create_individual(f"{NS}m{i}"), ad.create_individual(f"{NS}c{i}")))
    axs.add(ad.data_factory.getOWLDeclarationAxiom(ad.create_individual(NS + "lonely")))
    ad.owl_manager.addAxioms(ont, axs)
    path = tmp_path / "family_ontology.owl"
    ad.owl_manager.saveOntology(ont, FunctionalSyntaxDocumentFormat(),
                                IRI.create(java.io.File(str(path)).toURI()))
    return path


def _assertions(ontology):
    from org.semanticweb.owlapi.model import AxiomType
    classes = {(str(a.getIndividual().toStringID()), str(a.getClassExpression().asOWLClass()
                                                          .toStringID()))
               for a in ontology.getAxioms(AxiomType.CLASS_ASSERTION)}
    roles = {(str(a.getSubject().toStringID()), str(a.getProperty().asOWLObjectProperty()
                                                    .toStringID()),
              str(a.getObject().toStringID()))
             for a in ontology.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION)}
    return classes, roles


@pytest.fixture
def dataset(tmp_path):
    raw = _raw(tmp_path)
    return FaithELFamilyDataset(split_seed=1, train_fraction=0.6, data_root=str(tmp_path / "d"),
                                url=raw.as_uri())


def test_the_split_partitions_the_entailed_assertions(dataset):
    train_c, train_r = _assertions(dataset.ontology)
    test_c, test_r = _assertions(dataset.testing)
    # Mother(m_i) entails Parent(m_i) and Person(m_i): 15 class assertions, 5 role assertions
    assert len(train_c) + len(test_c) == 15 and not (train_c & test_c)
    assert len(train_r) + len(test_r) == 5 and not (train_r & test_r)
    assert len(train_c) == 9 and len(test_c) == 6 and len(train_r) == 3 and len(test_r) == 2
    assert {c for _, c in train_c | test_c} == {NS + "Mother", NS + "Parent", NS + "Person"}
    assert all(r == NS + "hasChild" for _, r, _ in train_r | test_r)
    # the TBox stays in the training ontology only; every individual is declared in both
    from org.semanticweb.owlapi.model import AxiomType
    assert dataset.ontology.getAxioms(AxiomType.SUBCLASS_OF).size() == 2
    assert dataset.testing.getAxioms(AxiomType.SUBCLASS_OF).size() == 0
    assert dataset.ontology.getIndividualsInSignature().size() == 11
    assert dataset.testing.getIndividualsInSignature().size() == 11


def test_the_split_is_deterministic_and_depends_on_the_seed(dataset, tmp_path):
    same = FaithELFamilyDataset(split_seed=1, train_fraction=0.6,
                                data_root=str(tmp_path / "d"), url="file:///nonexistent")
    assert _assertions(same.testing) == _assertions(dataset.testing)
    other = FaithELFamilyDataset(split_seed=2, train_fraction=0.6,
                                 data_root=str(tmp_path / "d"),
                                 url=pathlib.Path(tmp_path / "family_ontology.owl").as_uri())
    assert _assertions(other.testing) != _assertions(dataset.testing)
    with pytest.raises(ValueError):
        FaithELFamilyDataset(train_fraction=1.0, data_root=str(tmp_path / "d"))


class _Oracle:
    """Cost 0 for the assertions in ``good``, else 1 + (index / 100): known ranking."""

    axiom_scoring = "method"
    scored_forms = ("class_assertion", "object_property_assertion")
    device = "cpu"

    def __init__(self, dataset, good_classes, good_roles):
        names = sorted(str(i.toStringID()) for i in dataset.ontology.getIndividualsInSignature())
        self.individual_index_dict = {n: i for i, n in enumerate(names)}
        classes = sorted(str(c.toStringID()) for c in dataset.ontology.getClassesInSignature())
        self.class_index_dict = {n: i for i, n in enumerate(classes)}
        self.object_property_index_dict = {NS + "hasChild": 0}
        self.dataset = dataset
        self.good_c = {(self.individual_index_dict[a], self.class_index_dict[c])
                       for a, c in good_classes}
        self.good_r = {(self.individual_index_dict[a], 0, self.individual_index_dict[b])
                       for a, _, b in good_roles}

    def axiom_costs(self, gci_name, rows):
        good = self.good_c if gci_name == "class_assertion" else self.good_r
        return th.tensor([0.0 if tuple(r) in good else 1.0 + r[0] / 100.0
                          for r in rows.tolist()])


def test_hits_follow_the_known_ranking(dataset):
    test_c, test_r = _assertions(dataset.testing)
    # a perfect oracle: only the test assertions are cheap, so every query has rank 1
    oracle = _Oracle(dataset, test_c, test_r)
    out = AssertionQueryHitsRegime(hits=(1, 3)).evaluate(oracle, dataset)
    assert out["concept"]["hits@1"] == 1.0 and out["role"]["hits@1"] == 1.0
    assert out["concept"]["mrr"] == 1.0 and out["role"]["mrr"] == 1.0
    # an oracle that knows no test assertion: a query is a hit only if one of its answers
    # has a low individual index (cost 1 + index/100, ties broken by the index)
    blind = _Oracle(dataset, set(), set())
    names = {i: n for n, i in blind.individual_index_dict.items()}
    reg = AssertionQueryHitsRegime(hits=(1, 3))
    out = reg.evaluate(blind, dataset)
    assert out["concept"]["hits@1"] <= out["concept"]["hits@3"] <= 1.0
    assert reg.info["individuals"] == 11 and reg.info["scoring"] == "model"
    queries = {}
    for a, c in test_c:
        queries.setdefault(c, set()).add(blind.individual_index_dict[a])
    expected_rank = [min(ans) + 1 for ans in queries.values()]       # cost grows with index
    assert out["concept"]["hits@1"] == pytest.approx(
        sum(r <= 1 for r in expected_rank) / len(expected_rank))
    assert out["concept"]["mrr"] == pytest.approx(
        sum(1.0 / r for r in expected_rank) / len(expected_rank))
    assert reg.info["concept_queries"] == len(queries)
    assert names


def test_one_hit_per_query_not_per_answer(dataset):
    """Two answers of one query count once; the best-ranked answer decides."""
    train_c, train_r = _assertions(dataset.ontology)
    test_c, test_r = _assertions(dataset.testing)
    oracle = _Oracle(dataset, train_c, train_r)
    reg = AssertionQueryHitsRegime(hits=(1,))
    out = reg.evaluate(oracle, dataset)
    assert reg.info["concept_queries"] == len({c for _, c in test_c})
    assert reg.info["role_queries"] == len({(a, r) for a, r, _ in test_r})
    assert 0.0 <= out["concept"]["hits@1"] <= 1.0


def test_filtering_removes_other_known_answers(dataset):
    train_c, train_r = _assertions(dataset.ontology)
    test_c, test_r = _assertions(dataset.testing)
    # train answers are cheaper than the test answers: unfiltered they hide the test answer
    oracle = _Oracle(dataset, train_c, train_r)
    raw = AssertionQueryHitsRegime(hits=(1,), filtered=False).evaluate(oracle, dataset)
    filt = AssertionQueryHitsRegime(hits=(1,), filtered=True).evaluate(oracle, dataset)
    assert filt["concept"]["mrr"] >= raw["concept"]["mrr"]
    assert filt["concept"]["hits@1"] >= raw["concept"]["hits@1"]


def test_registration():
    bench = get_benchmark("faithel-family-assertions")
    assert bench.evaluation_regime == "assertion-query-hits"
    card = get_regime("assertion-query-hits")
    assert card.implementation.endswith("AssertionQueryHitsRegime")
    from mowl.benchmarks.recipes import list_recipes
    assert list_recipes("faithel-family-assertions", "faithel")
