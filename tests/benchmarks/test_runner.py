"""The evaluation regimes and the runner end to end on tiny ontologies (no downloads)."""

import json
import math
import os

import pytest

import mowl  # noqa: F401  (the tests package starts the JVM)
from mowl.benchmarks import Task
from mowl.benchmarks.spec import Benchmark, PublishedResult
from mowl.benchmarks.entries._refs import JACKERMEIER2024
from mowl.benchmarks.runner import (BenchmarkResult, UnsupportedModelError, compare,
                                    family_for, run_benchmark)
from tests.benchmarks.tiny import NS, scratch_dir, tiny_dataset

TINY_BENCH = Benchmark(
    key="tiny-prediction", name="tiny", task=Task.AXIOM_PREDICTION, description="tiny",
    dataset="tests.benchmarks.tiny.tiny_dataset",
    protocol="mowl.benchmarks.protocols.bindings.Box2ELPredictionProtocol",
    metrics=("hits@1", "hits@10", "hits@100", "med", "mrr", "mr", "auc"),
    subsets=("gci0", "gci1", "gci2", "gci3", "combined"), references=(JACKERMEIER2024,))


@pytest.fixture(scope="module")
def dataset():
    return tiny_dataset()


@pytest.fixture(scope="module")
def trained_elem(dataset):
    import torch as th
    from mowl.models.elembeddings.model import ELEmbeddings
    th.manual_seed(0)
    model = ELEmbeddings(dataset, embed_dim=4,
                         model_filepath=os.path.join(scratch_dir(), "elem.pt"))
    model.eval_gci_name = "gci0"
    model.train(epochs=3)
    return model


def _check(results, subsets, metrics):
    assert set(results) == set(subsets)
    for s in subsets:
        assert set(metrics) <= set(results[s])


def test_normal_form_ranking(trained_elem, dataset):
    from mowl.evaluation.normal_form_ranking import NormalFormRankingRegime
    regime = NormalFormRankingRegime()
    res = regime.evaluate(trained_elem, dataset)
    _check(res, regime.subsets, regime.metrics)
    assert regime.info["scoring"] == "centre_distance"
    n = regime.info["num_candidates"]
    for s in regime.subsets:
        assert 1 <= res[s]["mr"] <= n and 0 <= res[s]["auc"] <= 1
    # Each normal form of the tiny test set has at least one query.
    assert all(regime.info["queries"][nf]["queries"] >= 1
               for nf in ("gci0", "gci1", "gci2", "gci3"))
    loss = NormalFormRankingRegime(scoring="loss")
    loss.evaluate(trained_elem, dataset)
    assert loss.info["scoring"] == "loss"


def test_other_regimes(trained_elem, dataset):
    from mowl.evaluation.normal_form_ranking import (InferredSubsumptionRankingRegime,
                                                     SubsumeeRankingRegime)
    from mowl.evaluation.filtered_ranking import ppi_ranking, cate_ranking
    from mowl.evaluation.sampled_f1 import SampledF1Regime
    for regime in (InferredSubsumptionRankingRegime(), SubsumeeRankingRegime(),
                   ppi_ranking(NS + "r"), cate_ranking("gci0", candidates="classes"),
                   cate_ranking("gci0", candidates="classes", closure=True),
                   SampledF1Regime(num_positives=10, num_negatives=2),
                   SampledF1Regime(gci_name="gci2", num_positives=10, num_negatives=2)):
        res = regime.evaluate(trained_elem, dataset)
        _check(res, regime.subsets, regime.metrics)
    # Filtering can only improve ranks.
    regime = cate_ranking("gci0", candidates="classes")
    regime.raw = True
    from mowl.evaluation.filtered_ranking import FilteredRankingRegime
    both = FilteredRankingRegime(gci_name="gci0", candidates="classes",
                                 filter_by=("train", "closure"))
    res = both.evaluate(trained_elem, dataset)["all"]
    assert res["f_mr"] <= res["mr"]


def test_exact_model_on_t3():
    from mowl.datasets.builtin.helly import HellyTkDataset
    from mowl.evaluation.exact_model import ExactModelRegime
    from mowl.models.elbe.model import ELBE
    d = scratch_dir()
    ds = HellyTkDataset(3, directory=d)
    model = ELBE(ds, embed_dim=2, model_filepath=os.path.join(d, "elbe.pt"))
    model.train(epochs=2)
    regime = ExactModelRegime()
    res = regime.evaluate(model, ds)["all"]
    assert regime.info["helly_certificates"] >= 1
    assert regime.info["region_kind"] == "single box"
    assert 0 <= res["axiom_satisfaction"] <= 1
    # Helly: single boxes cannot be a non-degenerate model of T_3.
    assert res["exact_model"] == 0.0


def test_run_benchmark_end_to_end(dataset):
    kwargs = dict(method="elem", model_kwargs={"embed_dim": 4}, epochs=3, seed=7,
                  dataset=dataset, workdir=scratch_dir())
    r1 = run_benchmark(TINY_BENCH, **kwargs)
    r2 = run_benchmark(TINY_BENCH, **kwargs)
    assert r1.metrics == r2.metrics, "same seed must give the same result"
    assert set(r1.metrics) == set(TINY_BENCH.subsets)
    assert r1.evaluation_regime == "box2el-normal-form-ranking"
    assert r1.settings["embed_dim"] == 4
    assert r1.protocol_info["scoring"] == "centre_distance"
    assert r1.wall_time["total"] >= 0
    path = os.path.join(scratch_dir(), "r.json")
    r1.save(path)
    loaded = BenchmarkResult.load(path)
    assert loaded.metrics == json.loads(r1.to_json())["metrics"]

    pub = [PublishedResult(benchmark="box2el-galen-prediction", method="elem",
                           metrics={"mr": 10.0, "hits@10": 0.5}, source=JACKERMEIER2024,
                           locator="test", subset="combined",
                           settings={"embed_dim": 200,
                                     "training_regime": ("dele-all-normal-form-negatives",)})]
    loaded.benchmark = "box2el-galen-prediction"
    rows = compare(loaded, published=pub)
    assert {r["metric"] for r in rows} == {"mr", "hits@10"}
    mr = [r for r in rows if r["metric"] == "mr"][0]
    assert math.isclose(mr["difference"], loaded.metrics["combined"]["mr"] - 10.0)
    assert mr["settings_differences"] == {"embed_dim": (4, 200)}
    assert mr["regime_match"] is False


def test_run_benchmark_with_factory(dataset):
    from mowl.models.elbe.model import ELBE

    def factory(ds):
        return ELBE(ds, embed_dim=3, model_filepath=os.path.join(scratch_dir(), "e.pt"))
    res = run_benchmark(TINY_BENCH, model_factory=factory, epochs=2, dataset=dataset)
    assert res.method is None and res.settings["embed_dim"] == 3


def test_unsupported_families():
    from mowl.base_models.syntactic_model import SyntacticModel
    with pytest.raises(UnsupportedModelError):
        family_for(int)
    fam = family_for(SyntacticModel)
    with pytest.raises(UnsupportedModelError, match="syntactic"):
        fam.build(SyntacticModel, None, {}, "cpu", "x.pt")


def test_factory_with_unsupported_model(dataset):
    class Plain:
        pass
    with pytest.raises(Exception):
        run_benchmark(TINY_BENCH, model_factory=lambda ds: Plain(), epochs=1,
                      dataset=dataset)


def test_helly_benchmark_via_registry():
    res = run_benchmark("helly-tk", method="elbe", model_kwargs={"embed_dim": 2},
                        epochs=2, dataset_kwargs={"directory": scratch_dir()})
    assert set(res.metrics["all"]) == {"axiom_satisfaction", "is_model", "empty_fraction",
                                       "no_interior_fraction", "exact_model",
                                       "exact_model_nonempty", "helly_violation_rate"}
    assert compare(res) == []
