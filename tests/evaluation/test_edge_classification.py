"""Threshold-F1 classification of subsumptions and the WordNet noun hierarchy dataset."""

import os

import numpy as np
import pytest
import torch as th

from mowl.benchmarks import get_benchmark, published_results
from mowl.datasets.builtin.wordnet_closure import WordNetNounClosureDataset, synset_iri
from mowl.evaluation.edge_classification import EdgeClassificationF1Regime

# animal > mammal > {dog, cat}; animal > bird; closure adds dog/cat < animal
VOCAB = ["animal.n.01", "mammal.n.01", "dog.n.01", "cat.n.01", "bird.n.01", "rock.n.01"]
TRAIN = [(0, 1), (1, 2), (1, 3), (0, 4)]              # parent, child
VALID, VALID_NEG = [(0, 2)], [(2, 0), (4, 2), (5, 2)]
TEST, TEST_NEG = [(0, 3)], [(3, 0), (4, 3), (5, 3), (3, 2)]


def _release(root):
    raw = os.path.join(root, "wordnet_noun_closure", "raw")
    os.makedirs(raw)
    files = {"vocab": [(i, n) for i, n in enumerate(VOCAB)], "train_0percent": TRAIN,
             "valid": VALID, "valid_neg": VALID_NEG, "test": TEST, "test_neg": TEST_NEG}
    for name, rows in files.items():
        with open(os.path.join(raw, f"noun_closure.tsv.{name}"), "w") as f:
            f.writelines(f"{a}\t{b}\n" for a, b in rows)


@pytest.fixture
def dataset(tmp_path):
    _release(str(tmp_path))
    return WordNetNounClosureDataset(percent=0, data_root=str(tmp_path))


def _subclass_pairs(ontology):
    from org.semanticweb.owlapi.model import AxiomType
    return {(str(a.getSubClass().toStringID()), str(a.getSuperClass().toStringID()))
            for a in ontology.getAxioms(AxiomType.SUBCLASS_OF)}


def test_the_dataset_reads_an_edge_as_parent_then_child(dataset, tmp_path):
    iri = {n.split(".")[0]: synset_iri(n) for n in VOCAB}
    assert _subclass_pairs(dataset.ontology) == {
        (iri["mammal"], iri["animal"]), (iri["dog"], iri["mammal"]),
        (iri["cat"], iri["mammal"]), (iri["bird"], iri["animal"])}
    assert _subclass_pairs(dataset.validation) == {(iri["dog"], iri["animal"])}
    assert _subclass_pairs(dataset.testing) == {(iri["cat"], iri["animal"])}
    assert (iri["animal"], iri["cat"]) in dataset.testing_negatives
    assert len(dataset.validation_negatives) == 3 and len(dataset.testing_negatives) == 4
    # every synset is a class of the training ontology, also the isolated one
    assert dataset.ontology.getClassesInSignature().size() == len(VOCAB)
    flipped = WordNetNounClosureDataset(percent=0, data_root=str(tmp_path),
                                        reversed_order=True)
    assert (iri["animal"], iri["mammal"]) in _subclass_pairs(flipped.ontology)
    with pytest.raises(ValueError):
        WordNetNounClosureDataset(percent=30, data_root=str(tmp_path))


class _Oracle:
    """A 'model' whose cost of C ⊑ D is 0 if D is an ancestor of C and 1 otherwise."""

    axiom_scoring = "method"
    scored_forms = ("gci0",)
    device = "cpu"

    def __init__(self, dataset, wrong=()):
        names = sorted(str(c.toStringID())
                       for c in dataset.ontology.getClassesInSignature())
        self.class_index_dict = {n: i for i, n in enumerate(names)}
        self.object_property_index_dict, self.individual_index_dict = {}, {}
        self.dataset = dataset
        up = {synset_iri(VOCAB[c]): synset_iri(VOCAB[p]) for p, c in TRAIN}
        self.true = set()
        for c in names:
            d = up.get(c)
            while d is not None:
                self.true.add((self.class_index_dict[c], self.class_index_dict[d]))
                d = up.get(d)
        self.true -= {(self.class_index_dict[c], self.class_index_dict[d]) for c, d in wrong}

    def axiom_costs(self, gci_name, rows):
        assert gci_name == "gci0"
        return th.tensor([0.0 if tuple(r) in self.true else 1.0 for r in rows.tolist()])


def test_a_transitive_model_gets_f1_one_with_the_given_negatives(dataset):
    regime = EdgeClassificationF1Regime(negatives="given")
    result = regime.evaluate(_Oracle(dataset), dataset)["all"]
    assert result == {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    assert regime.info["negatives"] == "given"
    assert regime.info["test_positives"] == 1 and regime.info["test_negatives"] == 4
    assert regime.info["threshold"] == 0.0 and regime.key == "hierarchy-threshold-f1"


def test_a_missed_test_edge_costs_recall(dataset):
    cat, animal = synset_iri("cat.n.01"), synset_iri("animal.n.01")
    regime = EdgeClassificationF1Regime()
    result = regime.evaluate(_Oracle(dataset, wrong=[(cat, animal)]), dataset)["all"]
    assert result["recall"] == 0.0 and result["f1"] == 0.0


def test_sampled_negatives_exclude_known_axioms(dataset):
    regime = EdgeClassificationF1Regime(negatives="sampled", num_negatives=50, seed=1)
    model = _Oracle(dataset)
    result = regime.evaluate(model, dataset)["all"]
    assert regime.info["negatives"] == "sampled" and regime.info["test_negatives"] > 0
    assert 0.0 <= result["f1"] <= 1.0 and np.isfinite(result["precision"])
    index = model.class_index_dict
    known = {(index[synset_iri(VOCAB[c])], index[synset_iri(VOCAB[p])])
             for p, c in TRAIN + VALID + TEST}
    gen = th.Generator().manual_seed(1)
    rows = regime._sample(th.tensor(sorted(known)), known, len(index), gen)
    assert len(rows) > 0 and not (set(map(tuple, rows.tolist())) & known)
    assert all(c != d for c, d in rows.tolist())
    plain = type("Plain", (), {"ontology": dataset.ontology, "validation": dataset.validation,
                               "testing": dataset.testing})()
    with pytest.raises(ValueError, match="sampled"):
        EdgeClassificationF1Regime(negatives="given").evaluate(model, plain)


def test_the_benchmark_and_its_published_numbers_are_registered():
    b = get_benchmark("wordnet-noun-closure-10")
    assert b.evaluation_regime == "hierarchy-threshold-f1"
    cones = [r for r in published_results(benchmark=b.key, method="hyperbolic_cones")
             if r.source.key == "ganea2018"]
    assert {r.settings["embed_dim"]: r.metrics["f1"] for r in cones} == {5: 0.801, 10: 0.859}
    shadow = published_results(benchmark="wordnet-noun-closure-0", method="shadow_cones")
    assert max(r.metrics["f1"] for r in shadow) == 0.522


def test_an_order_model_trains_and_is_classified(dataset):
    from mowl.models import OrderEmbeddings
    th.manual_seed(0)
    model = OrderEmbeddings(dataset, embed_dim=4, learning_rate=0.05)
    model.train(epochs=300, validate_every=1000)
    result = EdgeClassificationF1Regime().evaluate(model, dataset)["all"]
    assert set(result) == {"f1", "precision", "recall"} and result["f1"] > 0.0


def test_kg_link_prediction_ranks_both_directions():
    from mowl.evaluation.filtered_ranking import KGLinkPredictionRegime
    from mowl.models import ELEmbeddings
    from tests.benchmarks.tiny import tiny_dataset
    data = tiny_dataset()
    model = ELEmbeddings(data, embed_dim=8)
    model.eval_gci_name = "gci2"
    model.train(epochs=2)
    regime = KGLinkPredictionRegime()
    out = regime.evaluate(model, data)
    assert set(out) == {"all", "tail", "head"} and regime.key == "kg-filtered-link-prediction"
    assert set(out["all"]) == {"f_mrr", "f_mr", "f_hits@1", "f_hits@3", "f_hits@10"}
    assert regime.info["directions"] == 2 and regime.info["queries"] >= 1
    low, high = sorted((out["tail"]["f_mrr"], out["head"]["f_mrr"]))
    assert low - 1e-9 <= out["all"]["f_mrr"] <= high + 1e-9


def test_kg_benchmarks_and_published_numbers_are_registered():
    b = get_benchmark("kg-fb15k237-link-prediction")
    assert b.evaluation_regime == "kg-filtered-link-prediction" and b.measures("link-prediction")
    boxe = {r.settings["negative_sampling"]: r.metrics["f_mrr"]
            for r in published_results(benchmark=b.key, method="boxe")}
    assert boxe == {"uniform": 0.318, "self-adversarial": 0.337}
    (base,) = [r for r in published_results(benchmark="kg-wn18rr-link-prediction",
                                            method="expressive")
               if r.settings["variant"] == "base"]
    assert base.metrics["f_hits@10"] == 0.597 and "f_mr" not in base.metrics
    from mowl.datasets.builtin.kg import KnowledgeGraphDataset
    with pytest.raises(ValueError):
        KnowledgeGraphDataset("NotAGraph")
