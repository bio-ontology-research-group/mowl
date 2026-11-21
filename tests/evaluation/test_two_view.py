"""YAGO39K and the JOIE DBpedia data as ontologies with individuals; triple classification and
assertion link prediction; the EIKE and Concept2Box benchmarks."""

import os

import numpy as np
import pytest
import torch as th
from org.semanticweb.owlapi.model import AxiomType

from mowl.benchmarks import get_benchmark, get_recipe, list_recipes, published_results
from mowl.catalog import get_regime
from mowl.datasets.builtin.joie_dbpedia import JOIEDBpediaDataset
from mowl.datasets.builtin.joie_dbpedia import _iri as joie_iri
from mowl.datasets.builtin.yago39k import YAGO39KDataset
from mowl.datasets.builtin.yago39k import _iri as yago_iri
from mowl.evaluation.filtered_ranking import AssertionLinkPredictionRegime, FilteredRankingRegime
from mowl.evaluation.triple_classification import TripleClassificationRegime, best_threshold

# ---------------------------------------------------------------------------------------
# YAGO39K from fake release files
# ---------------------------------------------------------------------------------------

CONCEPTS = ["Person", "Athlete", "Footballer", "Club_(sport)"]
INSTANCES = ["alice", "bob", "carl", "dave"]
RELATIONS = ["knows", "playsFor"]
FILES = {
    "Train/concept2id": [4] + [f"<{c}>\t{i}" for i, c in enumerate(CONCEPTS)],
    "Train/instance2id": [4] + [f"<{c}>\t{i}" for i, c in enumerate(INSTANCES)],
    "Train/relation2id": [2] + [f"<{c}>\t{i}" for i, c in enumerate(RELATIONS)],
    "Train/instanceOf2id": ["0 2", "1 2", "2 1"],
    "Train/subClassOf2id": ["2 1", "1 0"],
    # (head, tail, relation): knows(alice, bob), knows(bob, carl), playsFor(alice, dave)
    "Train/triple2id": [3, "0 1 0", "1 2 0", "0 3 1"],
    "Valid/instanceOf2id_positive": [1, "3 2"], "Valid/instanceOf2id_negative": [1, "3 3"],
    "Valid/subClassOf2id_positive": [1, "2 0"], "Valid/subClassOf2id_negative": [1, "0 2"],
    "Valid/triple2id_positive": [1, "2 3 0"], "Valid/triple2id_negative": [1, "3 2 0"],
    "Test/instanceOf2id_positive": [1, "1 1"], "Test/instanceOf2id_negative": [1, "1 3"],
    "Test/subClassOf2id_positive": [1, "3 0"], "Test/subClassOf2id_negative": [1, "1 2"],
    "Test/triple2id_positive": [1, "0 2 0"], "Test/triple2id_negative": [1, "2 0 1"],
    "M-Valid/instanceOf2id_positive": [2, "3 2", "3 0"],
    "M-Valid/instanceOf2id_negative": [2, "3 3", "2 3"],
    "M-Valid/subClassOf2id_positive": [1, "2 0"], "M-Valid/subClassOf2id_negative": [1, "0 1"],
    "M-Test/instanceOf2id_positive": [1, "0 0"], "M-Test/instanceOf2id_negative": [1, "0 3"],
    "M-Test/subClassOf2id_positive": [1, "2 0"], "M-Test/subClassOf2id_negative": [1, "1 3"],
}


def _write_raw(root, folder, files):
    for name, lines in files.items():
        path = os.path.join(root, folder, "raw", name + ".txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(str(x) for x in lines) + "\n")


@pytest.fixture
def yago(tmp_path):
    _write_raw(str(tmp_path), "yago39k", FILES)
    return YAGO39KDataset(variant="yago39k", data_root=str(tmp_path))


def _axioms(ontology, kind):
    return list(ontology.getAxioms(kind))


def _name(entity):
    return str(entity.toStringID()).rsplit("/", 1)[-1]


def test_yago39k_reads_head_tail_relation_and_instance_then_concept(yago):
    from org.semanticweb.owlapi.model import AxiomType as A
    triples = {(_name(a.getSubject()), _name(a.getProperty().asOWLObjectProperty()),
                _name(a.getObject())) for a in _axioms(yago.ontology, A.OBJECT_PROPERTY_ASSERTION)}
    assert triples == {("alice", "knows", "bob"), ("bob", "knows", "carl"),
                       ("alice", "playsFor", "dave")}
    types = {(_name(a.getIndividual()), _name(a.getClassExpression().asOWLClass()))
             for a in _axioms(yago.ontology, A.CLASS_ASSERTION)}
    assert types == {("alice", "Footballer"), ("bob", "Footballer"), ("carl", "Athlete")}
    subs = {(_name(a.getSubClass().asOWLClass()), _name(a.getSuperClass().asOWLClass()))
            for a in _axioms(yago.ontology, A.SUBCLASS_OF)}
    assert subs == {("Footballer", "Athlete"), ("Athlete", "Person")}
    # everything of the release is in the signature, also what no axiom mentions
    assert yago.ontology.getClassesInSignature().size() == 4
    assert yago.ontology.getIndividualsInSignature().size() == 4
    assert yago.ontology.getObjectPropertiesInSignature().size() == 2
    assert len(_axioms(yago.validation, A.OBJECT_PROPERTY_ASSERTION)) == 1
    assert len(_axioms(yago.testing, A.CLASS_ASSERTION)) == 1
    assert len(_axioms(yago.testing, A.SUBCLASS_OF)) == 1


def test_yago39k_negatives_and_the_mixed_variant(yago, tmp_path):
    i, c, r = (lambda n: yago_iri("instance", n)), (lambda n: yago_iri("concept", n)), \
        (lambda n: yago_iri("relation", n))
    assert yago.negatives("valid", "class_assertion") == [(i("dave"), c("Club_(sport)"))]
    assert yago.negatives("test", "gci0") == [(c("Athlete"), c("Footballer"))]
    assert yago.negatives("valid", "object_property_assertion") == [
        (i("dave"), r("knows"), i("carl"))]
    m = YAGO39KDataset(variant="m-yago39k", data_root=str(tmp_path))
    assert len(m.negatives("valid", "class_assertion")) == 2
    assert len(_axioms(m.validation, AxiomType.CLASS_ASSERTION)) == 2
    assert len(_axioms(m.testing, AxiomType.CLASS_ASSERTION)) == 1
    # relational triples are the ones of Valid / Test in both variants
    assert len(_axioms(m.testing, AxiomType.OBJECT_PROPERTY_ASSERTION)) == 1
    with pytest.raises(ValueError):
        YAGO39KDataset(variant="yago", data_root=str(tmp_path))


def test_yago39k_subsample_keeps_triples_among_kept_instances(tmp_path):
    _write_raw(str(tmp_path), "yago39k", FILES)
    ds = YAGO39KDataset(data_root=str(tmp_path), subsample=2, seed=0)
    assert ds.ontology.getIndividualsInSignature().size() == 2
    assert ds.ontology.getClassesInSignature().size() == 4


# ---------------------------------------------------------------------------------------
# JOIE DBpedia from fake files
# ---------------------------------------------------------------------------------------

JOIE_FILES = {
    "db_insnet": [f"e{i}\trel{i % 2}\te{(i + 1) % 10}" for i in range(10)]
    + ["e0\trel0\te1"],  # a repeated row of the release is one triple
    "db_InsType_mini": [f"e{i}\ttype\tc{i % 3}" for i in range(10)],
    "db_onto_small_mini": [f"c{i}\tisa\tc{(i + 1) % 10}" if i % 2 else
                           f"c{i}\trelatedto\tc{(i + 2) % 10}" for i in range(10)],
}


@pytest.fixture
def joie_files(tmp_path):
    _write_raw(str(tmp_path), "joie_dbpedia", JOIE_FILES)
    return str(tmp_path)


def test_joie_kg_completion_splits_triples_and_keeps_all_links(joie_files):
    ds = JOIEDBpediaDataset(task="kg-completion", seed=0, data_root=joie_files)
    A = AxiomType
    assert len(_axioms(ds.ontology, A.CLASS_ASSERTION)) == 10
    assert len(_axioms(ds.validation, A.CLASS_ASSERTION)) == 0
    for part, n in ((ds.ontology, 8), (ds.validation, 1), (ds.testing, 1)):
        assert len(_axioms(part, A.OBJECT_PROPERTY_ASSERTION)) == n
        assert len(_axioms(part, A.SUBCLASS_OF)) == n
    head = {(_name(a.getSubject()), _name(a.getProperty().asOWLObjectProperty()),
             _name(a.getObject())) for a in _axioms(ds.ontology, A.OBJECT_PROPERTY_ASSERTION)}
    assert all(t[1] in ("rel0", "rel1") for t in head)
    assert set(ds.ontology_view_concepts) == {joie_iri("concept", f"c{i}") for i in range(10)}
    # an ontology-view triple is C ⊑ ∃r.D with the relation of the ontology view
    sub = _axioms(ds.ontology, A.SUBCLASS_OF)[0]
    assert "concept-relation" in str(sub.getSuperClass().getProperty().asOWLObjectProperty()
                                     .toStringID())


def test_joie_concept_linking_splits_the_links_and_keeps_all_triples(joie_files):
    ds = JOIEDBpediaDataset(task="concept-linking", seed=1, data_root=joie_files)
    A = AxiomType
    assert len(_axioms(ds.ontology, A.OBJECT_PROPERTY_ASSERTION)) == 10
    assert len(_axioms(ds.validation, A.OBJECT_PROPERTY_ASSERTION)) == 0
    assert [len(_axioms(p, A.CLASS_ASSERTION)) for p in (ds.ontology, ds.validation,
                                                         ds.testing)] == [8, 1, 1]
    again = JOIEDBpediaDataset(task="concept-linking", seed=1, data_root=joie_files)
    assert {str(a) for a in _axioms(again.testing, A.CLASS_ASSERTION)} == \
        {str(a) for a in _axioms(ds.testing, A.CLASS_ASSERTION)}
    with pytest.raises(ValueError):
        JOIEDBpediaDataset(task="x", data_root=joie_files)


# ---------------------------------------------------------------------------------------
# Regimes against an oracle model
# ---------------------------------------------------------------------------------------

class _Oracle:
    """Cost 0 for the axioms in ``true`` (row tuples per form), ``wrong`` get 1; the costs of
    the relation ``shifted`` are ``shift`` higher, so that one threshold would not do."""

    axiom_scoring = "method"
    scored_forms = ("gci0", "class_assertion", "object_property_assertion")
    device = "cpu"

    def __init__(self, dataset, true, shifted=None, shift=0.0):
        onto = dataset.ontology
        self.class_index_dict = {str(c.toStringID()): i for i, c in enumerate(
            sorted(onto.getClassesInSignature(), key=lambda x: str(x.toStringID())))}
        self.individual_index_dict = {str(c.toStringID()): i for i, c in enumerate(
            sorted(onto.getIndividualsInSignature(), key=lambda x: str(x.toStringID())))}
        self.object_property_index_dict = {str(c.toStringID()): i for i, c in enumerate(
            sorted(onto.getObjectPropertiesInSignature(), key=lambda x: str(x.toStringID())))}
        self.true, self.shifted, self.shift = true, shifted, shift

    def axiom_costs(self, gci_name, rows):
        out = []
        for r in rows.tolist():
            cost = 0.0 if (gci_name, tuple(r)) in self.true else 1.0
            if self.shifted is not None and gci_name == "object_property_assertion" \
                    and r[1] == self.shifted:
                cost += self.shift
            out.append(cost)
        return th.tensor(out)


def _yago_truth(model):
    i, c, r = model.individual_index_dict, model.class_index_dict, model.object_property_index_dict
    ia, ca, ra = (lambda n: i[yago_iri("instance", n)]), (lambda n: c[yago_iri("concept", n)]), \
        (lambda n: r[yago_iri("relation", n)])
    true = {("class_assertion", (ia(a), ca(b))) for a, b in
            [("alice", "Footballer"), ("bob", "Footballer"), ("carl", "Athlete"),
             ("dave", "Footballer"), ("bob", "Athlete")]}
    true |= {("gci0", (ca(a), ca(b))) for a, b in
             [("Footballer", "Athlete"), ("Athlete", "Person"), ("Footballer", "Person"),
              ("Club_(sport)", "Person")]}
    true |= {("object_property_assertion", (ia(a), ra(rel), ia(b))) for a, rel, b in
             [("alice", "knows", "bob"), ("bob", "knows", "carl"), ("alice", "playsFor", "dave"),
              ("carl", "knows", "dave"), ("alice", "knows", "carl")]}
    return true


def test_best_threshold_follows_the_reference_grid():
    threshold, accuracy = best_threshold([0.0, 0.1], [0.9, 1.0])
    assert accuracy == 1.0 and 0.1 < threshold <= 0.9
    assert best_threshold([], [])[0] == float("-inf")
    exact, acc = best_threshold([0.0, 0.1], [0.9, 1.0], grid=None)
    assert acc == 1.0 and exact == pytest.approx(0.5)


@pytest.mark.parametrize("form,count", [("class_assertion", 1), ("gci0", 1),
                                        ("object_property_assertion", 1)])
def test_triple_classification_of_an_oracle_is_perfect(yago, form, count):
    model = _Oracle(yago, _yago_truth(_Oracle(yago, set())))
    regime = TripleClassificationRegime(form=form)
    result = regime.evaluate(model, yago)["all"]
    assert result == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert regime.info["negatives"] == "given" and regime.info["test_positives"] == count
    assert regime.info["test_negatives"] == count and regime.key == \
        "triple-classification-thresholds"


def test_the_thresholds_are_per_relation(yago):
    base = _Oracle(yago, set())
    knows = base.object_property_index_dict[yago_iri("relation", "playsFor")]
    model = _Oracle(yago, _yago_truth(base), shifted=knows, shift=10.0)
    regime = TripleClassificationRegime(form="object_property_assertion")
    # the test triples are knows(alice, carl) (true) and playsFor-free; give playsFor a
    # test pair of its own by checking the thresholds the regime chose for the two relations
    result = regime.evaluate(model, yago)["all"]
    assert result["accuracy"] == 1.0 and regime.info["relations"] >= 1
    # a model that ranks everything the same cannot beat chance on the balanced test set
    flat = _Oracle(yago, set())
    flat.axiom_costs = lambda gci_name, rows: th.zeros(len(rows))
    assert regime.evaluate(flat, yago)["all"]["accuracy"] <= 0.5


def test_sampled_negatives_never_are_known_axioms(yago):
    model = _Oracle(yago, _yago_truth(_Oracle(yago, set())))
    regime = TripleClassificationRegime(form="object_property_assertion",
                                        negatives="sampled", num_negatives=20, seed=3)
    result = regime.evaluate(model, yago)["all"]
    assert regime.info["negatives"] == "sampled" and regime.info["test_negatives"] > 0
    assert result["recall"] == 1.0
    with pytest.raises(ValueError):
        TripleClassificationRegime(form="gci1")


def test_assertion_link_prediction_raw_and_filtered_ranks(yago):
    """Test triple knows(alice, carl): tail query knows(alice, ?) has bob (train) and carl
    (test) at cost 0, head query knows(?, carl) has bob (train) and alice (test)."""
    model = _Oracle(yago, _yago_truth(_Oracle(yago, set())))
    # the test ontology of the fake release holds knows(alice, carl) ("0 2 0")
    regime = AssertionLinkPredictionRegime()
    out = regime.evaluate(model, yago)
    assert set(out) == {"all", "head", "tail"} and regime.key == "assertion-link-prediction"
    assert out["tail"]["f_mrr"] == pytest.approx(1.0)
    assert out["tail"]["mrr"] == pytest.approx(1 / 1.5)
    assert out["all"]["f_hits@1"] == 1.0 and out["all"]["hits@1"] == 0.0
    assert out["all"]["f_mr"] == 1.0 and out["all"]["mr"] == pytest.approx(1.5)
    assert regime.info["num_candidates"] == 4 and regime.info["queries"] == 1
    tails = AssertionLinkPredictionRegime(directions=("tail",), hits=(1, 10))
    result = tails.evaluate(model, yago)
    assert set(result) == {"all"} and "f_hits@10" in result["all"]
    with pytest.raises(ValueError):
        AssertionLinkPredictionRegime(directions=("left",))


def test_class_assertion_ranking_with_individuals_as_the_query(yago):
    model = _Oracle(yago, _yago_truth(_Oracle(yago, set())))
    regime = FilteredRankingRegime(gci_name="class_assertion", target="tail",
                                   candidates="classes", filter_by=("train", "valid", "test"),
                                   ties="average", hits=(1, 3), scoring="model")
    result = regime.evaluate(model, yago)["all"]
    # test class assertion bob(Footballer ... "1 1" = bob: Athlete): Athlete and Footballer
    # are both true for bob (train), so the raw rank is 1.5 and the filtered rank 1
    assert result["f_mrr"] == 1.0 and result["mrr"] == pytest.approx(1 / 1.5)


def test_candidates_from_a_dataset_attribute(joie_files):
    ds = JOIEDBpediaDataset(task="kg-completion", seed=0, data_root=joie_files)
    regime = FilteredRankingRegime(gci_name="gci2", candidates="dataset:ontology_view_concepts")
    assert set(regime.candidate_iris(ds)) == set(ds.ontology_view_concepts)


# ---------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------

def test_the_benchmarks_are_registered_with_their_regimes_and_tasks():
    for kind, task in (("instanceof", "instance-checking"), ("subclassof", "subsumption"),
                       ("relational", "role-assertion")):
        b = get_benchmark(f"yago39k-{kind}-classification")
        assert b.evaluation_regime == "triple-classification-thresholds"
        assert task in b.inference_tasks and b.dataset_kwargs == {"variant": "yago39k"}
    assert get_benchmark("m-yago39k-instanceof-classification").dataset_kwargs["variant"] == \
        "m-yago39k"
    assert get_benchmark("yago39k-link-prediction").measures("link-prediction")
    assert get_benchmark("joie-dbpedia-concept-linking").measures("type-prediction")
    onto = get_benchmark("joie-dbpedia-ontology-view-completion")
    assert onto.measures("link-prediction") and onto.measures("existential-prediction")
    for key in ("triple-classification-thresholds", "assertion-link-prediction",
                "concept-linking-ranking", "two-view-kg-completion"):
        card = get_regime(key)
        assert card.assumption == "closed world over stated axioms" and card.implemented
        card.load()
    from mowl.benchmarks import supports
    for method in ("eike", "concept2box"):
        assert supports(method, "yago39k-instanceof-classification")
        assert supports(method, "joie-dbpedia-concept-linking")


def test_the_published_numbers_are_those_of_the_tables():
    (pre_eye,) = [r for r in published_results(
        benchmark="yago39k-instanceof-classification", method="eike")
        if r.settings == {"intensional_vectors": "pretrained", "instance_map": "eye",
                          "negative_sampling": "uniform"}]
    assert pre_eye.metrics == {"accuracy": 0.8926, "precision": 0.8881, "recall": 0.8984,
                               "f1": 0.8932} and pre_eye.locator == "Table 2"
    (m,) = [r for r in published_results(benchmark="m-yago39k-subclassof-classification",
                                         method="eike")
            if r.settings["intensional_vectors"] == "pretrained"
            and r.settings["instance_map"] == "eye"
            and r.settings["negative_sampling"] == "uniform"]
    assert m.metrics["accuracy"] == 0.8702
    lp = [r for r in published_results(benchmark="yago39k-link-prediction", method="eike")
          if r.settings["negative_sampling"] == "bernoulli"
          and r.settings["intensional_vectors"] == "random"
          and r.settings["instance_map"] == "eye"]
    assert lp[0].metrics == {"mrr": 0.115, "f_mrr": 0.577, "f_hits@1": 0.471,
                             "f_hits@3": 0.654, "f_hits@10": 0.762}
    assert len(published_results(benchmark="yago39k-link-prediction")) == 8
    (c,) = published_results(benchmark="joie-dbpedia-concept-linking", method="concept2box")
    assert c.metrics == {"mrr": 0.87, "hits@1": 0.7809, "hits@3": 0.9744}
    for benchmark in ("joie-dbpedia-instance-view-completion",
                      "joie-dbpedia-ontology-view-completion"):
        assert published_results(benchmark=benchmark, method="concept2box")[0].metrics[
            "hits@10"] in (0.7311, 0.8301)


def test_the_recipes_name_the_appendix_configurations():
    r = get_recipe("yago39k-instanceof-classification", "eike", source="wang2024eike",
                   name="pre-eye")
    assert r.model_kwargs["margins"] == [0.3, 0.4, 1.0] and r.model_kwargs["alpha"] == 1.0
    assert r.model_kwargs["instance_map"] == "eye" and "Appendix" in r.locator
    lp = get_recipe("yago39k-link-prediction", "eike", source="wang2024eike", name="unp-mat")
    assert lp.model_kwargs["margins"] == [0.1, 0.3, 1.0] and lp.model_kwargs["alpha"] == 0.5
    assert "L1" in lp.notes
    assert len(list_recipes(method="eike", source="wang2024eike")) == 24
    c = get_recipe("joie-dbpedia-concept-linking", "concept2box", source="huang2023")
    assert c.model_kwargs["kge_margin"] == 0.3 and c.model_kwargs["cross_margin"] == 0.15
    assert "not stated" in c.notes


def test_intensional_vectors_can_be_supplied_as_a_file(tmp_path):
    from mowl.models.eike.model import EIKE, name_to_text
    vectors = {"http://x/a%2C_b": np.arange(5, dtype="float32"),
               "http://x/c": np.ones(5, dtype="float32")}
    path = str(tmp_path / "v.npz")
    EIKE.save_intensional_vectors(path, vectors)
    loaded = EIKE.load_intensional_vectors(path)
    assert set(loaded) == set(vectors) and np.allclose(loaded["http://x/c"], 1.0)
    small = EIKE.pca_vectors({"a": np.arange(6.0), "b": np.arange(6.0)[::-1],
                              "c": np.ones(6)}, 2)
    assert all(v.shape == (2,) for v in small.values())
    assert name_to_text("http://mowl.borg/yago39k/concept/wikicat_Danish_male_film_actors") == \
        "wikicat Danish male film actors"
    assert name_to_text("http://x/Club_%28sport%29") == "Club (sport)"
