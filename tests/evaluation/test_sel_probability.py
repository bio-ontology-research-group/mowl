"""Statistical EL conditionals, the PMP split and the interval metrics of BoxSEL."""

import numpy as np
import pytest
import torch as th

from mowl.benchmarks import get_benchmark, published_results
from mowl.benchmarks.recipes import list_recipes
from mowl.catalog import get_regime
from mowl.datasets.builtin.sel_conditionals import (SELConditionalDataset,
                                                    conditionals_from_assertions,
                                                    write_conditionals_file)
from mowl.evaluation.sel_probability import (SELProbabilityRegime, fixed_estimator,
                                             interval_metrics, random_estimator)

NS = "http://example.org/s#"
ROOT, A, B, C = NS + "Root", NS + "A", NS + "B", NS + "C"
R = NS + "r"


def _members():
    # 10 entities in Root; A: e0-e5; B: e3-e7; C: e0, e1
    e = [f"e{i}" for i in range(10)]
    return {ROOT: set(e), A: set(e[0:6]), B: set(e[3:8]), C: {"e0", "e1"}}


def _lookup(conds):
    return {(str(c[0]) if isinstance(c[0], str) else repr(c[0]),
             str(c[1]) if isinstance(c[1], str) else repr(c[1])): c[2] for c in conds}


def test_conditionals_follow_algorithm_1():
    members = _members()
    # r relates e0 -> e3 (B), e1 -> e3, e2 -> e9 (not in A), e8 -> e0 (A)
    pairs = {R: {("e0", "e3"), ("e1", "e3"), ("e2", "e9"), ("e8", "e0")}}
    conds = conditionals_from_assertions(members, pairs)
    look = _lookup(conds)
    assert look[(B, A)] == pytest.approx(3 / 6)                  # (B|A): e3, e4, e5 of 6
    assert look[(A, B)] == pytest.approx(3 / 5)
    assert look[(ROOT, A)] == 1.0 and look[(A, ROOT)] == 0.6
    assert look[(C, repr(["and", A, B]))] == 0.0                # A and B share e3-e5, none in C
    assert look[(C, repr(["and", A, ROOT]))] == pytest.approx(2 / 6)
    # (B | ∃r.A): entities with an r-successor in A: e0, e1 (-> e3) and e8 (-> e0); none in B
    assert look[(B, repr(["some", R, A]))] == 0.0
    # (∃r.B | A): A entities with an r-successor in B: e0, e1 of the 6
    assert look[(repr(["some", R, B]), A)] == pytest.approx(2 / 6)
    # conditionals with an empty premise are undefined and skipped
    empty = NS + "Empty"
    more = conditionals_from_assertions({**members, empty: set()}, pairs)
    assert not any(c[1] == empty or (isinstance(c[1], list) and empty in c[1]) for c in more)
    assert not any(isinstance(c[1], list) and c[1][0] == "some" and c[1][2] == empty
                   for c in more)
    assert any(c[0] == empty for c in more)                       # (Empty | A) = 0 is defined
    # unordered pairs once: (B | A ⊓ C) exists, (B | C ⊓ A) does not
    with_pair = [c for c in conds if c[0] == B and isinstance(c[1], list)]
    assert len([c for c in with_pair if set(c[1][1:]) == {A, C}]) == 1


def _file(tmp_path):
    members = _members()
    pairs = {R: {("e0", "e3"), ("e1", "e3"), ("e2", "e9"), ("e8", "e0")}}
    conds = conditionals_from_assertions(members, pairs)
    path = tmp_path / "toy.json"
    write_conditionals_file(str(path), conds, ROOT)
    return str(path), conds


def test_dataset_splits_the_root_conditionals_and_keeps_the_rest(tmp_path):
    path, conds = _file(tmp_path)
    data = SELConditionalDataset(path=path, query_fraction=0.5, split_seed=1,
                                 data_root=str(tmp_path))
    root_conds = [c for c in conds if c[1] == ROOT]
    assert len(data.queries) == round(0.5 * len(root_conds))
    assert all(q[1] == ROOT for q in data.queries)
    assert len(data.conditionals) + len(data.queries) == len(conds) == len(data.all_conditionals)
    # the learning set keeps every conditional that is not a held-out query
    held = {(repr(q[0]), repr(q[1])) for q in data.queries}
    assert not held & {(repr(c[0]), repr(c[1])) for c in data.conditionals}
    again = SELConditionalDataset(path=path, query_fraction=0.5, split_seed=1,
                                  data_root=str(tmp_path))
    assert again.queries == data.queries
    other = SELConditionalDataset(path=path, query_fraction=0.5, split_seed=2,
                                  data_root=str(tmp_path))
    assert other.queries != data.queries
    # the classes and the role are declared, nothing else
    assert {str(c.toStringID()) for c in data.ontology.getClassesInSignature()} >= {ROOT, A, B, C}
    assert [str(p.toStringID()) for p in data.ontology.getObjectPropertiesInSignature()] == [R]
    with pytest.raises(FileNotFoundError, match="not released"):
        SELConditionalDataset("person", data_root=str(tmp_path / "empty"))
    with pytest.raises(ValueError):
        SELConditionalDataset(path=path, query_fraction=1.0)


def test_pmp_interval_is_computed_from_the_two_premises(tmp_path):
    path = tmp_path / "pmp.json"
    conds = [(A, ROOT, 0.5), (B, ROOT, 0.4), (B, ["and", ROOT, A], 0.8),
             (B, ["and", ROOT, C], 0.2), (C, ROOT, 0.25)]
    write_conditionals_file(str(path), conds, ROOT)
    data = SELConditionalDataset(path=str(path), query_fraction=0.2, split_seed=0,
                                 data_root=str(tmp_path))
    # the only held-out query is chosen by the seed; force it to be (B | Root)
    data.queries = [(B, ROOT, 0.4, 0.4)]
    data.conditionals = [(A, ROOT, 0.5, 0.5), (B, ["and", ROOT, A], 0.8, 0.8),
                         (B, ["and", ROOT, C], 0.2, 0.2), (C, ROOT, 0.25, 0.25)]
    (query,) = data.pmp_queries()
    # via D = A: [0.5 * 0.8, min(1, 0.5 * 0.8 + 1 - 0.5)] = [0.4, 0.9]
    # via D = C: [0.25 * 0.2, min(1, 0.25 * 0.2 + 1 - 0.25)] = [0.05, 0.8]; the intersection
    assert query[0] == B and query[1] == ROOT
    assert query[2] == pytest.approx(0.4) and query[3] == pytest.approx(0.8)
    # a query without premises in the learning set is not evaluable
    data.queries = [(C, ROOT, 0.25, 0.25)]
    assert data.pmp_queries() == []


def test_interval_metrics_by_hand():
    truth = np.array([[0.4, 0.8], [0.2, 0.5], [0.0, 1.0], [0.3, 0.3]])
    lower = np.array([0.5, 0.1, 0.0, 0.35])
    upper = np.array([0.7, 0.4, 1.0, 0.35])
    m = interval_metrics(truth, lower, upper)
    # inside: query 1 yes, 2 no (lower 0.1 < 0.2), 3 yes, 4 no (0.35 > 0.3)
    assert m["soundness_accuracy"] == 0.5
    assert m["soundness_error"] == pytest.approx((0 + 0.1 + 0 + 0.05) / 4)
    assert m["approximation_gap"] == pytest.approx(
        ((0.1 + 0.1) + (0.1 + 0.1) + 0 + (0.05 + 0.05)) / 4)
    lo, up = fixed_estimator(4)
    assert interval_metrics(truth, lo, up)["soundness_accuracy"] == 0.25
    lo, up = random_estimator(1000, np.random.default_rng(0))
    assert (lo <= up).all() and (lo >= 0).all() and (up <= 1).all()


class _Oracle:
    """A 'model' with a fixed probability per (conclusion, premise) and no training."""

    device = "cpu"

    def __init__(self, table):
        self.table, self.ids, self.added = table, {}, None
        self.module = self

    def add_conditionals(self, conditionals):
        self.added = list(conditionals)

    def _compile(self, expr):
        k = repr(expr)
        return ("x",), self.ids.setdefault(k, len(self.ids))

    def probability(self, sd, sc, ld, lc):
        names = {i: k for k, i in self.ids.items()}
        return th.tensor([self.table.get((names[int(d)], names[int(c)]), 0.5)
                          for d, c in zip(ld, lc)])


def test_regime_scores_an_oracle_and_the_baselines(tmp_path):
    path = tmp_path / "pmp.json"
    conds = [(A, ROOT, 0.5), (B, ROOT, 0.4), (B, ["and", ROOT, A], 0.8)]
    write_conditionals_file(str(path), conds, ROOT)
    data = SELConditionalDataset(path=str(path), query_fraction=0.5, split_seed=0,
                                 data_root=str(tmp_path))
    data.queries = [(B, ROOT, 0.4, 0.4)]
    data.conditionals = [(A, ROOT, 0.5, 0.5), (B, ["and", ROOT, A], 0.8, 0.8)]
    # entailed interval of (B | Root): [0.4, 0.9]
    inside = _Oracle({(repr(B), repr(ROOT)): 0.6,
                      (repr(A), repr(ROOT)): 0.5, (repr(B), repr(["and", ROOT, A])): 0.7})
    regime = SELProbabilityRegime()
    regime.initialise_model(inside, data)
    assert inside.added == data.conditionals
    out = regime.evaluate(inside, data)["all"]
    assert out["soundness_accuracy"] == 1.0 and out["soundness_error"] == 0.0
    assert out["approximation_gap"] == pytest.approx(0.2 + 0.3)
    assert out["mean_absolute_error"] == pytest.approx((0 + 0.1) / 2)
    assert regime.info["queries"] == 1 and regime.info["baselines"]["fixed"][
        "soundness_accuracy"] == 0.0
    outside = _Oracle({(repr(B), repr(ROOT)): 0.95})
    out = regime.evaluate(outside, data)["all"]
    assert out["soundness_accuracy"] == 0.0 and out["soundness_error"] == pytest.approx(0.05)
    # two embeddings: the interval is [min, max] of the estimates
    ens = regime.evaluate_ensemble([inside, _Oracle({(repr(B), repr(ROOT)): 0.45})], data)
    assert ens["all"]["soundness_accuracy"] == 1.0
    assert ens["all"]["approximation_gap"] == pytest.approx(abs(0.4 - 0.45) + abs(0.9 - 0.6))
    assert regime.info["embeddings"] == 2
    with pytest.raises(TypeError, match="conditionals"):
        regime.initialise_model(object(), data)


def test_regime_runs_with_boxsel_on_a_tiny_knowledge_base(tmp_path):
    """The whole path with the real model on a small knowledge base (synthetic: random
    memberships, only to exercise the pipeline)."""
    from mowl.benchmarks.runner import run_benchmark
    rng = np.random.default_rng(0)
    entities = [f"e{i}" for i in range(60)]
    classes = [NS + f"K{i}" for i in range(5)]
    members = {c: {e for e in entities if rng.random() < 0.5} for c in classes}
    root = NS + "Top"
    members[root] = set(entities)
    path = tmp_path / "synthetic.json"
    write_conditionals_file(str(path), conditionals_from_assertions(members), root)
    data = SELConditionalDataset(path=str(path), data_root=str(tmp_path), split_seed=0)
    result = run_benchmark("boxsel-sel-person", method="boxsel", dataset=data, epochs=5,
                           model_kwargs={"embed_dim": 4, "learning_rate": 0.05},
                           workdir=str(tmp_path / "w"))
    metrics = result.metrics["all"]
    assert set(metrics) == {"soundness_accuracy", "soundness_error", "approximation_gap",
                            "mean_absolute_error"}
    assert result.protocol_info["queries"] > 0
    assert result.protocol_info["learning_conditionals"] == len(data.conditionals)


def test_benchmarks_published_numbers_and_recipes_register():
    for name in ("person", "country", "hybrid"):
        bench = get_benchmark(f"boxsel-sel-{name}")
        assert bench.evaluation_regime == "sel-pmp-intervals"
        assert "NOT RUNNABLE" in bench.notes
    assert get_regime("sel-pmp-intervals").implementation.endswith("SELProbabilityRegime")
    (row,) = published_results(benchmark="boxsel-sel-person", method="boxsel")
    assert row.metrics == {"soundness_accuracy": 0.893, "soundness_error": 0.013,
                           "approximation_gap": 0.233}
    (row,) = published_results(benchmark="boxsel-sel-hybrid", method="boxsel")
    assert row.metrics["approximation_gap"] == 0.293 and row.settings == {"n_embeddings": 60}
    (recipe,) = list_recipes(benchmark="boxsel-sel-country", method="boxsel")
    assert recipe.model_kwargs["embed_dim"] == 16 and recipe.model_kwargs["learning_rate"] == 0.05
