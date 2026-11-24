"""Implementations of a method in the benchmark runner: the registry, the baseline
implementation registered by mowl.baselines.benchmarks, the manifest generator and the
collector of the three-way table."""
import json
import os
import subprocess
import sys

import pytest

import mowl  # noqa: F401  (the tests package starts the JVM)
from mowl.benchmarks import Task
from mowl.benchmarks.entries._refs import JACKERMEIER2024
from mowl.benchmarks.implementations import (Implementation, PartsImplementation,
                                             get_implementation, list_implementations,
                                             register_implementation, with_run_seed)
from mowl.benchmarks.spec import Benchmark
from tests.benchmarks.tiny import scratch_dir

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

TINY = Benchmark(
    key="tiny-impl-prediction", name="tiny", task=Task.AXIOM_PREDICTION, description="tiny",
    dataset="tests.benchmarks.tiny.tiny_dataset",
    protocol="mowl.benchmarks.protocols.bindings.Box2ELPredictionProtocol",
    metrics=("hits@1", "hits@10", "med", "mrr", "mr", "auc"),
    subsets=("gci0", "gci1", "gci2", "gci3", "combined"), references=(JACKERMEIER2024,))


# -- the registry ---------------------------------------------------------------------------

def test_parts_is_the_default_and_unknown_names_say_how_to_register():
    assert isinstance(get_implementation(None), PartsImplementation)
    assert "parts" in list_implementations()
    with pytest.raises(KeyError, match="--plugin"):
        get_implementation("no-such-implementation")


def test_with_run_seed_hands_the_run_seed_to_explicit_seed_parameters():
    def fixed(dataset, seed=0):
        pass

    def free(dataset, seed=None):
        pass

    assert with_run_seed(fixed, {}, 3) == {"seed": 3}
    assert with_run_seed(fixed, {"seed": 7}, 3) == {"seed": 7}
    assert with_run_seed(free, {}, 3) == {"seed": 3}
    assert with_run_seed(lambda dataset, **kw: None, {"a": 1}, 3) == {"a": 1}


def test_regimes_built_by_the_runner_get_the_run_seed():
    from mowl.benchmarks.runner import resolve_training_regime
    regime = resolve_training_regime(("box2el-code-training", {"batch_size": 4}), seed=5)
    batches = [r for r in regime.regimes if type(r).__name__ == "PerNormalFormMiniBatches"]
    a = resolve_training_regime(("box2el-code-training", {"batch_size": 4}), seed=5)
    b = [r for r in a.regimes if type(r).__name__ == "PerNormalFormMiniBatches"]
    assert batches[0].rng.integers(1 << 30) == b[0].rng.integers(1 << 30)


def test_a_registered_implementation_is_run_and_recorded():
    from mowl.benchmarks.runner import run_benchmark

    class Recording(PartsImplementation):
        name = "recording-test"
        built = []

        def build(self, card, dataset, model_kwargs, device, model_filepath, seed=0):
            self.built.append((card.key, dict(model_kwargs), seed))
            return super().build(card, dataset, model_kwargs, device, model_filepath, seed)

        def describe(self, model):
            return {**super().describe(model), "note": "recorded"}

    impl = register_implementation(Recording())
    result = run_benchmark(TINY, method="elbe", model_kwargs={"embed_dim": 4}, epochs=1,
                           seed=2, implementation="recording-test", workdir=scratch_dir())
    assert impl.built == [("elbe", {"embed_dim": 4}, 2)]
    assert result.implementation["name"] == "recording-test"
    assert result.implementation["note"] == "recorded"
    assert result.options["seed"] == 2 and result.options["model_kwargs"] == {"embed_dim": 4}
    assert result.environment.get("git_commit")
    data = json.loads(result.to_json())
    assert data["implementation"]["model_class"] == "mowl.models.elbe.model.ELBE"


def test_an_implementation_may_refuse_a_method():
    from mowl.benchmarks.runner import UnsupportedModelError, run_benchmark

    class OnlyElbe(Implementation):
        name = "only-elbe-test"

        def supports(self, method):
            return method == "elbe"

    register_implementation(OnlyElbe())
    with pytest.raises(UnsupportedModelError, match="only-elbe-test"):
        run_benchmark(TINY, method="elem", implementation="only-elbe-test")


def test_plugins_register_benchmarks_and_recipes(monkeypatch):
    from dataclasses import replace
    from mowl.benchmarks import recipes, registry
    monkeypatch.setattr(registry, "_BENCHMARKS", dict(registry._benchmarks()))
    monkeypatch.setattr(recipes, "_RECIPES", recipes._recipes())
    registry.register_benchmark(TINY)
    registry.register_benchmark(TINY)          # the same benchmark again: no error
    assert registry.get_benchmark(TINY.key) is TINY
    with pytest.raises(ValueError):
        registry.register_benchmark(replace(TINY, name="other"))
    recipe = replace(recipes.list_recipes("box2el-galen-prediction", "elbe")[0],
                     benchmark=TINY.key)
    recipes.register_recipe(recipe)
    recipes.register_recipe(recipe)
    assert recipes.list_recipes(TINY.key) == [recipe]


# -- the baseline implementation --------------------------------------------------------------

def test_the_baseline_implementation_covers_every_compared_method():
    import mowl.baselines.benchmarks as baselines
    from tests.baselines.harness import all_pairs
    impl = get_implementation("baseline")
    assert impl is baselines.IMPLEMENTATION
    from mowl.catalog import list_methods
    catalogue = {c.key for c in list_methods(implemented=True)}
    # (kge_model, the generic PyKEEN wrapper, has a Pair but is not a catalogue method)
    missing = {p.method: impl.unsupported_reason(p.method) for p in all_pairs()
               if p.method in catalogue and not impl.supports(p.method)}
    assert not missing, missing
    for method in baselines.UNSUPPORTED:
        assert not impl.supports(method)


@pytest.mark.parametrize("method", ["elbe", "elem", "regd"])
def test_parts_and_baseline_give_the_same_metrics(method):
    """The same benchmark, regime, model arguments and seed: the same numbers. RegD has its
    loss as a regime of the method; the recipe's regime must not replace it."""
    import mowl.baselines.benchmarks  # noqa: F401
    from mowl.benchmarks.runner import run_benchmark
    out = {}
    for impl in ("parts", "baseline"):
        out[impl] = run_benchmark(
            TINY, method=method, model_kwargs={"embed_dim": 4}, epochs=3, seed=1,
            training_regime=("box2el-code-training", {"batch_size": 4, "validate_every": 1}),
            validate_every=1, implementation=impl, workdir=scratch_dir())
    assert out["baseline"].implementation["model_class"].startswith("mowl.baselines.")
    assert out["parts"].implementation["model_class"].startswith("mowl.models.")
    assert out["parts"].metrics == out["baseline"].metrics
    assert out["parts"].training_regime == out["baseline"].training_regime
    if method == "regd":
        assert "GroupedLogSumExpContrastiveLoss" in \
            out["parts"].implementation["method_regimes_kept"]


def test_the_baseline_refuses_an_option_it_does_not_have():
    import mowl.baselines.benchmarks  # noqa: F401
    from mowl.benchmarks.runner import UnsupportedModelError, run_benchmark
    with pytest.raises(UnsupportedModelError, match="exists only in the combination"):
        run_benchmark(TINY, method="elbe", model_kwargs={"no_such_option": 1}, epochs=1,
                      implementation="baseline", workdir=scratch_dir())


def test_the_library_module_entry_runs_the_baseline():
    d = scratch_dir()
    out = os.path.join(d, "r.json")
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "mowl.baselines.benchmarks", "run",
         "toy-box2el-galen-prediction", "--plugin", "tests.benchmarks.toy_benchmarks",
         "--method", "elbe", "--epochs", "1", "--recipe", "jackermeier2024",
         "--param", "embed_dim=2", "--jvm-memory", "1g", "--workdir", d, "--out", out],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.load(open(out))
    assert data["implementation"]["name"] == "baseline"
    assert data["implementation"]["model_class"] == "mowl.baselines.models.elbe.model.ELBE"
    assert data["recipe"]["source"] == "jackermeier2024"
    assert data["training_regime"] == "box2el-code-training"


# -- the manifest generator -----------------------------------------------------------------

@pytest.fixture(scope="module")
def manifests():
    import benchmark_manifests as bm
    repro, repro_skipped = bm.reproduction_runs([0, 1])
    common, common_skipped = bm.common_runs([0], baseline_methods=("elbe",))
    return bm, repro, repro_skipped, common, common_skipped


def test_manifest_tags_are_unique_and_encode_the_run(manifests):
    bm, repro, _, common, _ = manifests
    runs = repro + common
    bm.check_unique(runs)
    for run in runs[:50]:
        tag = bm.tag(run)
        assert run.benchmark in tag and run.method in tag and run.implementation in tag
        assert tag.endswith(f"s{run.seed}") and "/" not in tag and ":" not in tag
        fields = bm.line(run).split("\t")
        assert len(fields) == 6 and fields[1] == run.benchmark and fields[2] == run.method
        assert "--recipe " in fields[5]
        assert ("--implementation baseline" in fields[5]) == (run.implementation == "baseline")


def test_every_compared_method_with_a_reproduction_recipe_runs_in_both_implementations(
        manifests):
    from mowl.benchmarks.recipes import list_recipes
    from mowl.benchmarks.registry import published_results
    from tests.baselines.harness import all_pairs
    _, repro, skipped, _, _ = manifests
    compared = {p.method for p in all_pairs()}
    published = {(p.benchmark, p.method, p.source.key) for p in published_results()}
    expected = {(r.benchmark, r.method) for r in list_recipes()
                if (r.benchmark, r.method, r.source.key) in published
                and r.method in compared}
    assert expected
    have = {}
    for run in repro:
        have.setdefault((run.benchmark, run.method), set()).add(run.implementation)
    for key in expected:
        assert have.get(key) == {"parts", "baseline"}, (key, have.get(key))
    assert {r.seed for r in repro} == {0, 1}


def test_the_joint_benchmarks_have_every_runnable_method(manifests):
    from mowl.benchmarks.comparison import runnable_methods
    bm, _, _, common, skipped = manifests
    benchmarks = bm.common_benchmarks()
    assert benchmarks and all(b.startswith("common-") for b in benchmarks)
    for bench in benchmarks:
        methods = {r.method for r in common if r.benchmark == bench}
        runnable, _ = runnable_methods(bench)
        not_run = {m for (b, m, _, i) in skipped if b == bench and i == "parts"}
        assert methods <= set(runnable)
        assert set(runnable) <= methods | not_run
        baseline = {r.method for r in common if r.benchmark == bench
                    and r.implementation == "baseline"}
        assert baseline <= {"elbe"}


def test_cost_classes():
    import benchmark_manifests as bm
    assert bm.cost_class("dfalc-revision-family-mask20", "dfalc", 200) == "fast"
    assert bm.cost_class("box2el-go-prediction", "elbe", 10000) == "slow"
    assert bm.cost_class("box2el-galen-prediction", "elbe", 5000) == "medium"
    assert bm.cost_class("ont-galen-inference", "ont", 1) == "slow"


def test_manifest_files_and_summary(tmp_path):
    import benchmark_manifests as bm
    assert bm.main(["--out-dir", str(tmp_path), "--methods", "elbe,dfalc",
                    "--families", "reproduction"]) == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "summary.txt" in files and any(f.startswith("repro-") for f in files)
    lines = [line for f in files if f.startswith("repro-")
             for line in open(tmp_path / f)]
    assert lines and all(line.count("\t") == 5 for line in lines)


# -- the collector --------------------------------------------------------------------------

def _result(benchmark, method, recipe, impl, seed, metrics):
    return {"benchmark": benchmark, "method": method, "seed": seed, "metrics": metrics,
            "implementation": {"name": impl},
            "recipe": None if recipe is None else {"source": recipe, "name": ""}}


def test_collector_three_way_rows_and_flags():
    from mowl.benchmarks.collect import collect, three_way_table, to_csv_string
    from mowl.benchmarks.registry import published_results
    pub = [p for p in published_results("box2el-galen-prediction", "box2el")
           if p.source.key == "jackermeier2024"][0]
    value = pub.metrics["hits@100"]
    sub = pub.subset
    results = [
        _result("box2el-galen-prediction", "box2el", "jackermeier2024", "parts", 0,
                {sub: {"hits@100": value}}),
        _result("box2el-galen-prediction", "box2el", "jackermeier2024", "baseline", 0,
                {sub: {"hits@100": value * 0.9}}),
    ]
    manifest = [{"tag": "t-parts", "benchmark": "box2el-galen-prediction", "method": "box2el",
                 "recipe": "jackermeier2024", "implementation": "parts"},
                {"tag": "t-base", "benchmark": "box2el-galen-prediction", "method": "box2el",
                 "recipe": "jackermeier2024", "implementation": "baseline"},
                {"tag": "t-elbe", "benchmark": "box2el-galen-prediction", "method": "elbe",
                 "recipe": "jackermeier2024", "implementation": "baseline"}]
    status = {"t-parts": "done", "t-base": "done", "t-elbe": "failed"}
    rows = collect(results, manifest, status)
    row = [r for r in rows if r["method"] == "box2el" and r["metric"] == "hits@100"
           and r["subset"] == sub][0]
    assert row["published"] == value and row["parts"] == value
    assert row["parts − baseline"] == pytest.approx(0.1 * value)
    assert "parts≠baseline" in row["flags"] and "baseline far from published" in row["flags"]
    assert "parts far from published" not in row["flags"]
    assert any("failed baseline" in r["flags"] for r in rows if r["method"] == "elbe")
    for fmt in ("text", "markdown", "rst"):
        assert "box2el" in three_way_table(rows, fmt)
    assert "parts − baseline" in to_csv_string(rows).splitlines()[0]


def test_collector_uses_the_seed_spread_and_reports_missing_runs():
    from mowl.benchmarks.collect import collect
    bench = "common-galen-subsumption-prediction"
    results = [_result(bench, "elbe", "mowl-common", "parts", s, {"all": {"f_mrr": v}})
               for s, v in enumerate((0.30, 0.34, 0.26))]
    results += [_result(bench, "elbe", "mowl-common", "baseline", s, {"all": {"f_mrr": v}})
                for s, v in enumerate((0.31, 0.33, 0.29))]
    rows = collect(results)
    row = [r for r in rows if r["metric"] == "f_mrr"][0]
    assert row["seeds"] == {"parts": 3, "baseline": 3}
    assert row["published"] is None and "parts≠baseline" not in row["flags"]
    # a tolerance of zero and one seed: any difference is flagged
    rows = collect(results[:1] + results[3:4], rel_tolerance=0)
    assert "parts≠baseline" in [r for r in rows if r["metric"] == "f_mrr"][0]["flags"]
    manifest = [{"tag": "x", "benchmark": bench, "method": "elem", "recipe": "mowl-common",
                 "implementation": "parts"}]
    rows = collect([], manifest, {"x": "missing"})
    assert rows and rows[0]["flags"] == "missing parts"


def test_collector_reads_manifests(tmp_path):
    from mowl.benchmarks.collect import read_manifest, status_of_manifest
    path = tmp_path / "m.tsv"
    path.write_text("a\tb1\tm1\t2\t0\t--recipe src:name --plugin x --implementation "
                    "baseline\nb\tb1\tm1\t2\t1\t--recipe src:name\n")
    runs = read_manifest(str(path))
    assert [(r["recipe"], r["implementation"], r["seed"]) for r in runs] == [
        ("src:name", "baseline", 0), ("src:name", "parts", 1)]
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "b.log").write_text("error")
    assert status_of_manifest(runs, str(tmp_path), str(tmp_path / "logs")) == {
        "a": "done", "b": "failed"}


def test_the_baseline_trains_with_the_direct_regime_where_the_library_one_needs_parts():
    """A recipe regime that switches the objective of a geometry built from parts cannot
    train a direct model; the baseline implementation uses the direct regime of the same
    name instead, and leaves every other regime class alone."""
    import importlib
    from mowl.baselines.benchmarks import BaselineImplementation, DIRECT_REGIMES
    from mowl.benchmarks.implementations import Implementation
    from mowl.benchmarks.runner import resolve_training_regime
    from mowl.training.dele import DELECodeLosses
    from mowl.training.batching import PerNormalFormMiniBatches

    baseline = BaselineImplementation()
    direct = baseline.regime_class(DELECodeLosses)
    assert direct.__module__ == "mowl.baselines.training.dele"
    assert direct.__name__ == "DELECodeLosses"
    assert baseline.regime_class(PerNormalFormMiniBatches) is PerNormalFormMiniBatches
    assert Implementation().regime_class(DELECodeLosses) is DELECodeLosses
    for library, frozen in DIRECT_REGIMES.items():
        importlib.import_module(library)
        importlib.import_module(frozen)
    regime = resolve_training_regime("dele-code-training", seed=0, implementation=baseline)
    assert type(regime).__module__ == "mowl.baselines.training.dele"
    assert type(resolve_training_regime("dele-code-training", seed=0)).__module__ \
        == "mowl.training.dele"
    # the regimes of the language-model reference codes accept the models of their own side
    for key in ("hit-code-training", "ont-code-training"):
        assert type(resolve_training_regime(key, seed=0, implementation=baseline)).__module__ \
            == "mowl.baselines.training.lm_reference"


def test_the_baseline_modules_are_ranked_by_the_same_published_scores():
    """The evaluation looks a module's published score up by class path; the direct
    modules must be registered, or they are silently ranked by their loss."""
    from mowl.baselines.benchmarks import register_scorers
    from mowl.evaluation import el_scoring
    register_scorers()   # what the baseline implementation does when it builds a model
    from mowl.baselines.nn.el.boxsquaredel.module import BoxSquaredELModule
    from mowl.baselines.nn.el.elem.module import ELEmModule
    from mowl.baselines.nn.el.emelpp.module import EmELPPModule

    class Model:
        def __init__(self, module):
            self.module = module

    table = el_scoring.CENTRE_DISTANCE_SCORERS
    for library_path, scorer in list(table.items()):
        if library_path.startswith("mowl.nn.") and "regd" not in library_path:
            assert table.get("mowl.baselines." + library_path[5:]) is scorer, library_path
    for cls, key in ((BoxSquaredELModule, "boxsquaredel.module.BoxSquaredELModule"),
                     (ELEmModule, "elem.module.ELEmModule"),
                     (EmELPPModule, "elem.module.ELEmModule")):
        module = cls.__new__(cls)
        assert el_scoring._scorer_of(Model(module)) is table["mowl.nn.el." + key], cls


def test_the_evaluations_on_regions_read_the_regions_of_the_direct_modules():
    """The sampled F1 of RegD is defined on the class regions; a trained direct module is
    read as the balls or boxes over its own tables."""
    import torch as th
    from mowl.baselines import benchmarks as plugin
    from mowl.baselines.nn.el.elbe.module import ELBEModule
    from mowl.baselines.nn.el.elem.module import ELEmModule
    from mowl.baselines.nn.el.regd.module import RegDELBEModule, RegDELEmModule
    from mowl.evaluation.sampled_f1 import class_regions
    plugin.register_region_readers()
    for cls in (ELEmModule, ELBEModule, RegDELBEModule, RegDELEmModule):
        module = cls(7, 2, 0, embed_dim=4)
        centers, sizes, is_ball = class_regions(module)
        assert th.equal(centers, module.class_embed.weight)
        table = module.class_rad if is_ball else module.class_offset
        assert th.allclose(sizes.reshape(7, -1), table.weight.abs().reshape(7, -1))
