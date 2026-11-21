"""Released query datasets (Query2Box / BetaE format): dataset, regime, training on the
released queries, benchmarks, published numbers and recipes. Built from tiny fake files."""

import os
import pickle
from collections import defaultdict

import numpy as np
import pytest
import torch as th

from mowl.benchmarks import get_benchmark, list_benchmarks, list_recipes, published_results
from mowl.datasets.builtin.query_sets import QueryAnsweringDataset, _SafeUnpickler
from mowl.evaluation import (LogicalQueryAnswering, QUERY_STRUCTURES, ReleasedQueryAnswering,
                             answer_query)
from mowl.models import AConE, BetaE, ConE, Query2Box
from mowl.models.query.model import QueryEmbeddingModel

# Six entities, two relations a (ids 0, 1) and b (ids 2, 3); odd ids are the inverses.
ENT = ["e0", "e1", "e2", "e3", "e4", "e5"]
REL = {0: "+a", 1: "-a", 2: "+b", 3: "-b"}
TRAIN = [(0, 0, 1), (1, 2, 2), (0, 0, 3), (4, 0, 1)]
VALID = [(0, 0, 2), (4, 0, 2), (3, 2, 5)]
TEST = [(1, 2, 4), (0, 0, 5), (4, 0, 5), (3, 2, 4)]
UNIVERSE = set(range(6))


def _both(triples):
    return sorted(set(triples) | {(t, r + 1, h) for h, r, t in triples})


def _graph(*parts):
    return np.array(_both([x for p in parts for x in p]))


def _write(directory, name, obj):
    with open(os.path.join(directory, name), "wb") as f:
        pickle.dump(obj, f)


def _answers(structure, query, graph):
    return answer_query(structure, query, graph, UNIVERSE)


QUERIES = {
    "1p": [(0, (0,))],
    "2p": [(0, (0, 2))],
    "2i": [((0, (0,)), (4, (0,)))],
    "2in": [((0, (0,)), (4, (2, -2)))],
    "pni": [((1, (2, 2, -2)), (4, (0,)))],
    "2u": [((0, (0,)), (3, (2,)), (-1,))],
}


def _fake_release(directory):
    os.makedirs(directory)
    for part, triples in (("train", TRAIN), ("valid", VALID), ("test", TEST)):
        with open(os.path.join(directory, f"{part}.txt"), "w") as f:
            f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in _both(triples))
    with open(os.path.join(directory, "stats.txt"), "w") as f:
        f.write("numentity: 6\nnumrelations: 4")
    _write(directory, "id2ent.pkl", dict(enumerate(ENT)))
    _write(directory, "id2rel.pkl", REL)
    g_train, g_valid, g_test = (_graph(TRAIN), _graph(TRAIN, VALID), _graph(TRAIN, VALID, TEST))
    by_structure = lambda: {QUERY_STRUCTURES[n]: set(qs) for n, qs in QUERIES.items()}  # noqa
    train_answers = defaultdict(set)
    for n, qs in QUERIES.items():
        for q in qs:
            train_answers[q] = _answers(QUERY_STRUCTURES[n], q, g_train)
    _write(directory, "train-queries.pkl", defaultdict(set, by_structure()))
    _write(directory, "train-answers.pkl", train_answers)
    for split, known, full in (("valid", g_train, g_valid), ("test", g_valid, g_test)):
        easy, hard = defaultdict(set), defaultdict(set)
        for n, qs in QUERIES.items():
            for q in qs:
                e = _answers(QUERY_STRUCTURES[n], q, known)
                easy[q] = e
                hard[q] = _answers(QUERY_STRUCTURES[n], q, full) - e
        _write(directory, f"{split}-queries.pkl", defaultdict(set, by_structure()))
        _write(directory, f"{split}-easy-answers.pkl", easy)
        _write(directory, f"{split}-hard-answers.pkl", hard)


@pytest.fixture
def release(tmp_path):
    _fake_release(str(tmp_path / "release"))
    return str(tmp_path / "release")


@pytest.fixture
def dataset(tmp_path, release):
    return QueryAnsweringDataset(name="Fake", data_root=str(tmp_path / "root"),
                                 data_dir=release)


def _identity(dataset):
    ent = {iri: i for i, iri in enumerate(dataset.entity_iris)}
    return ent.__getitem__, lambda iri, inverse: (2 * dataset.relation_iris.index(iri)
                                                  + (1 if inverse else 0))


def test_the_training_graph_is_the_forward_edges(dataset):
    from org.semanticweb.owlapi.model import AxiomType
    axioms = {str(a.getSubClass().toStringID()): str(a.getSuperClass()) for a in
              dataset.ontology.getAxioms(AxiomType.SUBCLASS_OF)}
    assert len(dataset.ontology.getAxioms(AxiomType.SUBCLASS_OF)) == len(TRAIN)   # no inverses
    e = QueryAnsweringDataset.entity_iri
    assert set(axioms) == {e("Fake", ENT[h]) for h, _, _ in TRAIN}
    assert len(dataset.ontology.getClassesInSignature()) == 6
    assert len(dataset.validation.getAxioms(AxiomType.SUBCLASS_OF)) == len(VALID)
    assert len(dataset.testing.getAxioms(AxiomType.SUBCLASS_OF)) == len(TEST)
    assert dataset.relation_iris == [QueryAnsweringDataset.relation_iri("Fake", "a"),
                                     QueryAnsweringDataset.relation_iri("Fake", "b")]


def test_validation_and_test_queries_have_the_easy_and_hard_answers(dataset):
    ent, rel = _identity(dataset)
    for split, known, full in (("valid", _graph(TRAIN), _graph(TRAIN, VALID)),
                               ("test", _graph(TRAIN, VALID), _graph(TRAIN, VALID, TEST))):
        queries = dataset.queries(split, ent, rel)
        assert set(queries) == set(QUERIES)
        for name, items in queries.items():
            ((q, easy, hard),) = items
            assert q == QUERIES[name][0]                       # identity mapping: same ids
            assert easy == _answers(QUERY_STRUCTURES[name], q, known)
            assert easy | hard == _answers(QUERY_STRUCTURES[name], q, full)
            assert not easy & hard
    # every query has a hard answer; the one of the 2p test query is e4 (edges (1, b, 4) and
    # (3, b, 4) are test edges)
    assert all(hard for items in dataset.queries("test", ent, rel).values()
               for _, _, hard in items)
    assert dataset.queries("test", ent, rel, names=["2p"])["2p"][0][2] == {4}
    assert dataset.queries("test", ent, rel, names=["1p"], limit=1)["1p"][0][1] == {1, 2, 3}


def test_ids_are_mapped_to_the_model(dataset):
    shifted = {iri: 10 + 2 * i for i, iri in enumerate(dataset.entity_iris)}
    relation = lambda iri, inverse: 100 + 2 * dataset.relation_iris.index(iri) + inverse   # noqa
    q = dataset.queries("test", shifted.__getitem__, relation, names=["2in", "2u", "pni"])
    ((query, easy, hard),) = q["2in"]
    assert query == ((10, (100,)), (18, (102, -2)))            # -2 (negation) is kept
    assert all(a % 2 == 0 and a >= 10 for a in easy | hard)
    assert q["2u"][0][0] == ((10, (100,)), (16, (102,)), (-1,))
    assert q["pni"][0][0] == ((12, (102, 102, -2)), (18, (100,)))
    assert dataset.queries("test", shifted.__getitem__, relation,
                           names=["2in"])["2in"] == q["2in"]


def test_inverse_relations_of_the_model_are_used_for_odd_ids(dataset):
    ent, rel = _identity(dataset)
    odd = QueryAnsweringDataset._map_rows(
        QUERY_STRUCTURES["1p"], np.array([[3, 1]]), *dataset._maps(ent, rel))
    assert odd.tolist() == [[3, 1]]
    assert dataset._maps(ent, rel)[1].tolist() == [0, 1, 2, 3]


def test_training_queries_and_their_answers(dataset):
    ent, rel = _identity(dataset)
    train = dataset.training_queries(ent, rel)
    assert set(train) == set(QUERIES)
    q = train["2in"]
    assert len(q) == 1 and q.rows.tolist() == [[0, 0, 4, 2, -2]]
    assert sorted(q.answer_set(0).tolist()) == [1, 3] and q.count(0) == 2
    assert sorted(train["1p"].answer_set(0).tolist()) == [1, 3]


def test_negation_is_absence_in_the_stated_graph(dataset):
    """2in: e4 has no b edge in the graph, so ¬(4, b) is every entity (closed world) and the
    query has the answers of its first atom."""
    ent, rel = _identity(dataset)
    ((q, easy, hard),) = dataset.queries("valid", ent, rel, names=["2in"])["2in"]
    assert easy == {1, 3} and hard == {2}


def test_the_pickles_are_read_without_running_code(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned"),))
    path = tmp_path / "evil.pkl"
    with open(path, "wb") as f:
        pickle.dump(Evil(), f)
    with open(path, "rb") as f, pytest.raises(pickle.UnpicklingError):
        _SafeUnpickler(f).load()
    assert not (tmp_path / "pwned").exists()


def test_unknown_names_and_missing_data_are_refused(tmp_path):
    with pytest.raises(ValueError, match="data_dir"):
        QueryAnsweringDataset(name="WN18RR-QA", data_root=str(tmp_path))
    with pytest.raises(ValueError, match="variant"):
        QueryAnsweringDataset(name="FB15k", variant="x", data_root=str(tmp_path))


def test_conversion_is_cached_and_atomic(tmp_path, release):
    root = str(tmp_path / "root")
    QueryAnsweringDataset(name="Fake", data_root=root, data_dir=release)
    files = sorted(os.listdir(os.path.join(root, "query", "Fake")))
    assert not [f for f in files if f.endswith(".part")]
    os.remove(os.path.join(release, "train-queries.pkl"))       # not needed the second time
    QueryAnsweringDataset(name="Fake", data_root=root, data_dir=release)


# -- the regime ----------------------------------------------------------------------------

def test_answers_of_negation_structures():
    g = np.array(_both(TRAIN))
    s = QUERY_STRUCTURES
    assert _answers(s["2in"], ((0, (0,)), (4, (0, -2))), g) == {3}   # {1, 3} minus {1}
    assert _answers(s["pni"], ((0, (0, 2, -2)), (4, (0,))), g) == {1}   # not {2}, and 4 -a-> 1
    # De Morgan form of a union gives the answers of the union
    dm = (((0, (0, -2)), (1, (2, -2))), (-2,))
    assert _answers(s["2u-DM"], dm, g) == _answers(s["2u"], ((0, (0,)), (1, (2,)), (-1,)), g)


class _Oracle:
    """Scores 1 for the answers on a graph."""

    def __init__(self, dataset, graph):
        self.graph, self.dataset = graph, dataset
        self.ids = {iri: i for i, iri in enumerate(dataset.entity_iris)}
        self.module = type("M", (), {"supports_negation": True})()

    def entity_id(self, iri):
        return self.ids[iri]

    def relation_id(self, iri, inverse):
        return 2 * self.dataset.relation_iris.index(iri) + (1 if inverse else 0)

    def score_query(self, structure, queries):
        out = np.zeros((len(queries), 6))
        for i, q in enumerate(queries):
            out[i, sorted(answer_query(structure, q, self.graph, UNIVERSE))] = 1.0
        return out


def test_released_regime_with_an_oracle_and_groups(dataset):
    regime = ReleasedQueryAnswering(split="test", ties="pessimistic")
    perfect = regime.evaluate(_Oracle(dataset, _graph(TRAIN, VALID, TEST)), dataset)
    assert perfect["2p"]["mrr"] == 1.0
    assert "average_epfo" in perfect and "average_negation" in perfect
    assert set(perfect) >= set(QUERIES) | {"average", "average_epfo", "average_negation"}
    mean = np.mean([perfect[n]["mrr"] for n in ("1p", "2p", "2i", "2u")])
    assert perfect["average_epfo"]["mrr"] == pytest.approx(mean)
    # an oracle of the known graph cannot find the hard answer: it is ranked below all
    blind = regime.evaluate(_Oracle(dataset, _graph(TRAIN, VALID)), dataset)
    assert blind["2p"]["mrr"] < 1.0
    assert regime.info["queries"]["2p"] == 1


def test_a_model_without_negation_skips_those_structures(dataset):
    model = _Oracle(dataset, _graph(TRAIN, VALID, TEST))
    model.module.supports_negation = False
    regime = ReleasedQueryAnswering(split="test")
    result = regime.evaluate(model, dataset)
    assert "2in" not in result and "average_negation" not in result
    assert regime.info["skipped_structures"] == ["2in", "pni"]


def test_vectorised_ranks_match_the_definition():
    regime = LogicalQueryAnswering({"1p": []})
    scores = np.array([0.5, 0.9, 0.5, 0.1, 0.9, 0.5])
    ranks = regime._ranks(scores, np.array([0, 2]), {0, 1, 2})      # others: 3, 4, 5
    # among {0.1, 0.9, 0.5} and the answer 0.5: one greater (0.9), one equal (0.5)
    assert ranks.tolist() == [2.5, 2.5]
    regime = LogicalQueryAnswering({"1p": []}, ties="optimistic")
    assert regime._ranks(scores, np.array([0]), {0, 1, 2}).tolist() == [2.0]


# -- training on the released queries ------------------------------------------------------

@pytest.mark.parametrize("cls,kwargs", [(Query2Box, {}), (BetaE, {}), (ConE, {}), (AConE, {})])
def test_models_train_on_the_released_queries(dataset, cls, kwargs):
    th.manual_seed(0)
    m = cls(dataset, embed_dim=8, batch_size=8, negatives=4, queries="released",
            hierarchy_closure=False, extension=False, steps_per_epoch=3,
            **({"structures": ["1p", "2p", "2i"]} if cls is Query2Box else {}))
    assert set(m.released) == ({"1p", "2p", "2i"} if cls is Query2Box
                               else {"1p", "2p", "2i", "2in", "pni"})
    m.train(epochs=2)
    assert len(m.loss_history) == 2 and all(np.isfinite(m.loss_history))
    result = ReleasedQueryAnswering(split="test", batch_size=4).evaluate(m, dataset)
    assert 0 < result["average"]["mrr"] <= 1
    assert ("2in" in result) == (cls is not Query2Box)


def test_released_training_rows_match_the_embedding_order(dataset):
    """The columns of a released row are the ids of the query in depth-first order."""
    from mowl.nn.query.expression import from_nested, ids
    m = BetaE(dataset, embed_dim=4, queries="released", hierarchy_closure=False,
              extension=False)
    for name, q in m.released.items():
        node = from_nested(q.structure, _unflatten(q.structure, q.rows[0]))
        assert ids(node) == q.rows[0][m._released_columns[name]].tolist()


def _unflatten(structure, row):
    from mowl.datasets.builtin.query_sets import _build
    return _build(structure, iter(row.tolist()))


def test_negatives_are_not_answers_and_weights_follow_the_answer_count(dataset):
    m = ConE(dataset, embed_dim=4, batch_size=64, negatives=8, queries="released",
             hierarchy_closure=False, extension=False)
    batch = m._released_batch("path")
    sig, rows, pos, neg, w = batch["1p"]
    for r, p, n, weight in zip(rows.tolist(), pos.squeeze(1).tolist(), neg.tolist(),
                               w.tolist()):
        answers = set(m.released["1p"].answer_set(0).tolist())
        assert p in answers and not set(n) & answers
        assert weight == pytest.approx(1 / np.sqrt(len(answers)))


def test_released_training_needs_a_query_dataset():
    from tests.benchmarks.tiny import tiny_dataset
    with pytest.raises(ValueError, match="training_queries"):
        ConE(tiny_dataset(), embed_dim=4, queries="released")
    with pytest.raises(ValueError, match="queries must be"):
        ConE(tiny_dataset(), embed_dim=4, queries="x")


# -- benchmarks, published numbers, recipes ------------------------------------------------

def test_benchmarks_are_registered_with_their_regime():
    keys = {b.key for b in list_benchmarks() if b.key.startswith("query-")}
    assert keys == {f"query-{kg}-{v}" for kg in ("fb15k", "fb15k237", "nell995")
                    for v in ("betae", "q2b")} | {"query-wn18rr-qa"}
    b = get_benchmark("query-fb15k237-betae")
    assert b.evaluation_regime == "query-answering"
    assert b.assumption.value == "closed world over stated axioms"
    assert b.inference_tasks == ("query-answering",)
    assert {"2in", "pni", "2u", "up", "average_epfo", "average_negation"} <= set(b.subsets)
    assert "negation as absence" in b.notes
    assert "2in" not in get_benchmark("query-fb15k237-q2b").subsets
    from mowl.benchmarks import methods_for
    assert {"query2box", "betae", "cone_qe", "acone"} <= {m.key for m in methods_for(b)}


def test_transcribed_published_numbers():
    def get(bench, method, subset, source=None, metric="mrr"):
        (r,) = [r for r in published_results(bench, method, subset)
                if metric in r.metrics and (source is None or r.source.key == source)]
        return r.metrics[metric]
    assert get("query-fb15k237-betae", "cone_qe", "1p") == 0.418
    assert get("query-fb15k237-betae", "cone_qe", "average_epfo") == 0.234
    assert get("query-nell995-betae", "betae", "inp", "ren2020betae") == 0.100
    assert get("query-nell995-betae", "acone", "3i") == 0.530
    assert get("query-nell995-betae", "acone", "pi") == 0.261
    assert get("query-fb15k-betae", "betae", "pi", "ren2020betae") == 0.439
    assert get("query-fb15k-betae", "betae", "2in", "ren2020betae", "hits@10") == 0.308
    assert get("query-fb15k237-q2b", "query2box", "ip", "ren2020q2b") == 0.105
    assert get("query-fb15k237-q2b", "query2box", "pi", "ren2020q2b") == 0.180
    assert get("query-fb15k237-q2b", "query2box", "average_epfo", "ren2020q2b",
               "hits@3") == 0.268
    assert get("query-wn18rr-qa", "acone", "3i") == 0.890
    assert get("query-wn18rr-qa", "cone_qe", "pni") == 0.076
    methods = {r.method for r in published_results() if r.benchmark.startswith("query-")}
    assert methods == {"query2box", "betae", "cone_qe", "acone"}


def test_every_published_subset_is_a_subset_of_its_benchmark():
    for r in published_results():
        if r.benchmark.startswith("query-"):
            assert r.subset in get_benchmark(r.benchmark).subsets, (r.benchmark, r.subset)


def test_recipes_train_on_the_released_queries_and_name_their_source():
    recipes = [r for r in list_recipes() if r.benchmark.startswith("query-")]
    assert len(recipes) == 17
    for r in recipes:
        assert r.model_kwargs["queries"] == "released"
        assert r.locator and r.notes
        get_benchmark(r.benchmark)
        epochs = r.epochs
        milestones = [epochs // 2, 3 * epochs // 4]
        assert r.regime_spec() == ("step-lr-decay", {"milestones": milestones, "gamma": 0.2})
    cone = [r for r in recipes if r.method == "cone_qe"]
    assert {r.benchmark: r.model_kwargs["learning_rate"] for r in cone} == {
        "query-fb15k-betae": 5e-5, "query-fb15k237-betae": 5e-5, "query-nell995-betae": 1e-4}
    assert [r.model_kwargs["structures"] for r in recipes
            if r.method == "query2box"][0] == ["1p", "2p", "3p", "2i", "3i"]


@pytest.mark.parametrize("method", ["cone_qe", "query2box"])
def test_a_recipe_runs_through_the_benchmark_runner(dataset, tmp_path, method):
    """run_benchmark with a recipe (released training queries, step decay) and the
    benchmark's protocol on the fake release; the published numbers are compared."""
    from mowl.benchmarks.runner import compare, run_benchmark
    result = run_benchmark(
        "query-fb15k237-betae", method=method, dataset=dataset, recipe=True, epochs=2,
        model_kwargs={"embed_dim": 8, "batch_size": 8, "negatives": 4, "steps_per_epoch": 2,
                      **({"hidden": 8, "layers": 1} if method == "cone_qe" else {})},
        workdir=str(tmp_path / "w"), device="cpu")
    assert result.training_regime == "step-lr-decay"
    assert result.settings["queries"] == "released"
    assert result.protocol_info["queries"]["1p"] == 1
    assert "average_epfo" in result.metrics and 0 < result.metrics["1p"]["mrr"] <= 1
    assert ("2in" in result.metrics) == (method != "query2box")
    rows = compare(result)
    assert rows and {r["method"] for r in rows} == {method}


def test_recipe_arguments_are_arguments_of_the_model_or_its_geometry():
    import inspect
    from mowl.catalog import get_method
    for r in list_recipes():
        if not r.benchmark.startswith("query-"):
            continue
        cls = get_method(r.method).load()
        allowed = (set(inspect.signature(QueryEmbeddingModel.__init__).parameters)
                   | set(inspect.signature(cls.geometry_class.__init__).parameters))
        assert set(r.model_kwargs) <= allowed, (r.benchmark, r.method,
                                                set(r.model_kwargs) - allowed)
