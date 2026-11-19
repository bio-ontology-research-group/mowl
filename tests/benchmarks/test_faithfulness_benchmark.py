"""Registry entries, published results, regime cards and protocol of the faithfulness
benchmarks (no network access)."""

import dataclasses
import importlib

import pytest

from mowl.benchmarks import Task, get_benchmark, list_benchmarks, published_results
from mowl.catalog.card import RegimeKind
from mowl.catalog.registry import get_regime

KEYS = ("faithfulness-pizza", "faithfulness-goplus")


@pytest.mark.parametrize("key", KEYS)
def test_registry_entry(key):
    b = get_benchmark(key)
    assert b.task is Task.FAITHFULNESS
    assert b in list_benchmarks(task="faithfulness")
    assert set(b.subsets) == {"gci0", "gci1", "gci2", "gci3", "gci1_bot"}
    module, _, cls = b.dataset.rpartition(".")
    assert hasattr(importlib.import_module(module), cls)
    module, _, cls = b.protocol.rpartition(".")
    assert hasattr(importlib.import_module(module), cls)
    assert set(b.higher_is_better) == set(b.metrics)


def test_published_results():
    res = published_results(benchmark="faithfulness-pizza", subset="gci0")
    assert {r.method for r in res} == {"elem", "emelpp", "box2el", "transe", "distmult",
                                       "complex", "boxe"}
    box2el = published_results(benchmark="faithfulness-pizza", method="box2el",
                               subset="gci0")[0]
    assert box2el.metrics["auc_pm"] == 0.74 and box2el.metrics["pair_fvr"] == 0.60
    assert box2el.locator.startswith("Table 4")
    go = published_results(benchmark="faithfulness-goplus", subset="gci2")
    assert all(set(r.metrics) == {"rho_p", "mrr", "hits@10"} for r in go)
    # Tables 5 and 7 (seven models), 9, 11 and 13 (three models).
    assert len(published_results(benchmark="faithfulness-goplus")) == 7 + 7 + 3 * 3
    for r in published_results(benchmark="faithfulness-pizza") + \
            published_results(benchmark="faithfulness-goplus"):
        assert set(r.metrics) <= set(get_benchmark(r.benchmark).metrics)


@pytest.mark.parametrize("key,impl", [
    ("faithfulness-hierarchy", "mowl.evaluation.faithfulness.FaithfulnessHierarchy"),
    ("modelness", "mowl.evaluation.faithfulness.LossModelness"),
    ("kb-embedding-properties", "mowl.evaluation.properties.KBEmbeddingProperties"),
])
def test_regime_cards(key, impl):
    card = get_regime(key)
    assert card.kind is RegimeKind.EVALUATION and card.implementation == impl


def test_protocol_runs_on_tiny_ontology():
    import mowl
    mowl.init_jvm("2g")
    from tests.evaluation.test_faithfulness import _university_dataset
    from mowl.evaluation.faithfulness import LogicalStatusOracle, oracle_scorer

    bench = dataclasses.replace(get_benchmark("faithfulness-pizza"),
                                protocol_kwargs={"n_per_class": 4, "count_models": True,
                                                 "k": 10})
    protocol = bench.load_protocol()
    assert protocol.key == "faithfulness-hierarchy"
    assert protocol.subsets == ("gci0",)
    ds = _university_dataset()
    oracle = LogicalStatusOracle.from_dataset(ds)
    res = protocol.evaluate(oracle_scorer(oracle), ds)
    assert "gci0" in res and set(res["gci0"]) == set(bench.metrics)
    assert res["gci0"]["auc_pm"] == 1.0 and res["gci0"]["fvr"] == 0.0
    assert set(res) == set(protocol.subsets)
