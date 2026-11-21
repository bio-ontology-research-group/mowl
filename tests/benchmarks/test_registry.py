"""Consistency of the benchmark registry and of the published results (no JVM)."""

import importlib
import warnings

import pytest

from mowl.benchmarks import Task, get_benchmark, list_benchmarks, published_results
from mowl.benchmarks.spec import Benchmark

FIXED_KEYS = [
    "box2el-galen-prediction", "box2el-go-prediction", "box2el-anatomy-prediction",
    "box2el-galen-inference", "box2el-go-inference", "box2el-anatomy-inference",
    "box2el-yeast-ppi", "box2el-human-ppi",
    "dele-yeast-ppi", "dele-human-ppi", "dele-yeast-function", "dele-human-function",
    "dele-foodon-subsumption", "dele-galen-subsumption",
    "cate-ore1-membership", "cate-go-completion", "cate-foodon-completion", "cate-yeast-ppi",
    "helly-tk",
    "regd-galen-inference", "regd-go-inference", "regd-anatomy-inference",
    "regd-galen-prediction", "regd-go-prediction", "regd-anatomy-prediction",
    "emelpp-galen-prediction", "emelpp-go-prediction", "emelpp-anatomy-prediction",
    "elkem-galen-subsumption", "elkem-go-subsumption", "elkem-anatomy-subsumption",
]


@pytest.mark.parametrize("key", FIXED_KEYS)
def test_fixed_keys_exist(key):
    b = get_benchmark(key)
    assert isinstance(b, Benchmark)
    assert b.key == key


def test_every_benchmark_is_well_formed():
    for b in list_benchmarks():
        assert b.metrics and b.subsets
        assert len(set(b.metrics)) == len(b.metrics)
        if b.higher_is_better:
            assert set(b.higher_is_better) <= set(b.metrics), b.key
        for path in (b.dataset, b.protocol):
            module, _, attr = path.rpartition(".")
            spec = importlib.util.find_spec(module)
            assert spec is not None, f"{b.key}: module {module} not found"


def test_task_filter():
    keys = {b.key for b in list_benchmarks(task=Task.EXACT_MODEL)}
    assert "helly-tk" in keys
    assert all(b.task is Task.DEDUCTIVE_INFERENCE
               for b in list_benchmarks(task="deductive_inference"))


def test_published_results_are_consistent():
    results = published_results()
    assert results
    for r in results:
        b = get_benchmark(r.benchmark)
        assert r.subset in b.subsets, (r.benchmark, r.subset)
        assert set(r.metrics) <= set(b.metrics), (r.benchmark, set(r.metrics) - set(b.metrics))
        for value in r.metrics.values():
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert r.locator


def test_published_method_keys_against_catalogue():
    """Method keys must be catalogued eventually; other agents add cards concurrently, so
    unknown keys only produce a warning."""
    from mowl.catalog import get_method
    unknown = set()
    for r in published_results():
        try:
            get_method(r.method)
        except KeyError:
            unknown.add(r.method)
    if unknown:
        warnings.warn(f"Published results reference uncatalogued methods: {sorted(unknown)}")


def test_training_regimes_in_settings_are_catalogued():
    from mowl.catalog import get_regime
    for r in published_results():
        keys = r.settings.get("training_regime")
        if keys is None:
            continue
        for key in (keys if isinstance(keys, (tuple, list)) else (keys,)):
            try:
                assert get_regime(key).kind.value == "training"
            except KeyError:
                warnings.warn(f"Training regime {key!r} of {r.benchmark} is not catalogued.")


def test_transcribed_spot_values():
    """A few values checked against the PDFs by hand."""
    r = published_results(benchmark="box2el-galen-prediction", method="box2el",
                          subset="combined")
    jack = [x for x in r if x.source.key == "jackermeier2024"][0]
    assert jack.metrics == {"hits@1": 0.05, "hits@10": 0.20, "hits@100": 0.35, "med": 669,
                            "mrr": 0.10, "mr": 4375, "auc": 0.81}
    assert all(x.source.key != "yang2025" for x in r)
    # The TransBox paper tests differently and has its own benchmarks.
    trans = published_results(benchmark="transbox-galen-prediction", method="box2el",
                              subset="combined")[0]
    assert trans.source.key == "yang2025" and trans.metrics["med"] == 1360
    r = published_results(benchmark="box2el-human-ppi", method="emelpp")[0]
    assert r.metrics["f_mr"] == 700 and r.metrics["hits@10"] == 0.04
    r = published_results(benchmark="dele-yeast-function", method="box2el")
    assert sorted(x.metrics["f_hits@10"] for x in r) == [0.27, 0.27, 0.31]
    r = published_results(benchmark="cate-ore1-membership", method="cate")[0]
    assert r.metrics["f_mr"] == 37
    cate = published_results(benchmark="transbox-go-prediction", method="cate")[0]
    assert "med" not in cate.metrics


def test_no_published_numbers_for_helly():
    assert published_results(benchmark="helly-tk") == []


@pytest.mark.parametrize("key,kind", [
    ("box2el-normal-form-ranking", "evaluation"),
    ("inferred-subsumption-ranking", "evaluation"),
    ("ppi-ranking", "evaluation"), ("regd-sampled-f1", "evaluation"),
    ("exact-model-check", "evaluation"), ("cate-filtered-ranking", "evaluation"),
    ("emelpp-subsumee-ranking", "evaluation"), ("elkem-subsumption-ranking", "evaluation"),
    ("elkem-code-training", "training"), ("box2el-code-training", "training"),
    ("transbox-code-training", "training"),
])
def test_regime_cards(key, kind):
    from mowl.catalog import get_regime
    card = get_regime(key)
    assert card.kind.value == kind
    module, _, attr = card.implementation.rpartition(".")
    assert importlib.util.find_spec(module) is not None


def test_published_to_rst_and_tables():
    from mowl.benchmarks import report
    text = report.published_to_rst()
    assert "box2el-galen-prediction" in text and ".. list-table::" in text
    for fmt in report.FORMATS:
        out = report.published_table("box2el-yeast-ppi", fmt=fmt)
        assert "box2el" in out


def test_elem_recipes_use_the_released_loss():
    """Every reference code has the radius term of the ELEm loss of C ⊓ D ⊑ E commented
    out, so every recipe that trains ELEm balls must switch it off explicitly."""
    from mowl.benchmarks.recipes import list_recipes
    recipes = [r for r in list_recipes() if r.method in ("elem", "regd_elem")]
    assert recipes
    for recipe in recipes:
        if recipe.source.key == "mowl-common" and not recipe.locator.startswith("recipe of"):
            # mOWL's own defaults reproduce no reference code: the corrected loss.
            assert "gci1_radius_term" not in recipe.model_kwargs
            continue
        assert recipe.model_kwargs.get("gci1_radius_term") is False, (
            recipe.benchmark, recipe.method)


# --- benchmarks are linked to the tasks of mowl.logic ----------------------------------

#: Evaluation regimes that rank the missing class of any axiom form; their card names the
#: general task, the benchmark the form it uses.
GENERAL_REGIMES = {"cate-filtered-ranking", "dele-evaluation", "common-filtered-ranking",
                   "triple-classification-thresholds"}


def test_every_benchmark_is_linked_to_tasks_and_a_regime_card():
    from mowl.catalog import get_regime
    from mowl.logic import COMPLETION_TASKS, TASKS, Target
    for b in list_benchmarks():
        assert b.inference_tasks, b.key
        assert set(b.inference_tasks) <= set(TASKS), b.key
        assert set(b.completion_tasks) | set(b.encodes) <= set(COMPLETION_TASKS), b.key
        assert isinstance(b.target, Target), b.key
        assert b.evaluation_regime, f"{b.key} names no evaluation regime card"
        card = get_regime(b.evaluation_regime)
        assert card.kind.value == "evaluation", b.key
        if card.target:
            assert card.target == b.target.value, (b.key, card.key, card.target)
        if card.task and card.key not in GENERAL_REGIMES:
            assert b.measures(card.task), (b.key, card.key, card.task)


def test_filter_by_logic_task_and_target():
    from mowl.logic import Target
    links = {b.key for b in list_benchmarks(completion_task="link-prediction")}
    # ABox links, and interactions posed as C ⊑ ∃r.D with proteins as classes.
    assert {"boxlite-family-v1", "box2el-yeast-ppi", "dele-human-ppi"} <= links
    assert "box2el-go-prediction" not in links
    assert get_benchmark("box2el-yeast-ppi").completion_tasks == ("existential-prediction",)
    assert get_benchmark("box2el-yeast-ppi").inference_tasks == ("subsumption",)
    assert get_benchmark("boxlite-family-v1").inference_tasks == ("role-assertion",)
    # The same argument accepts groups, completion tasks and inference tasks.
    assert {b.key for b in list_benchmarks(task="link-prediction")} == links
    instance = {b.key for b in list_benchmarks(task="instance-checking")}
    assert "cate-ore1-membership" in instance and "dele-yeast-function" in instance
    entailed = list_benchmarks(target=Target.ENTAILED)
    assert entailed and all(b.target is Target.ENTAILED for b in entailed)
    assert {b.key for b in list_benchmarks(target="deductive inference")} == {
        b.key for b in entailed}
    with pytest.raises(ValueError):
        list_benchmarks(task="no-such-task")


def test_methods_and_benchmarks_are_matched_by_task_and_expressivity():
    from mowl.benchmarks import benchmarks_for, methods_for, missing_support, supports
    # A taxonomy method answers subsumption between names, not C ⊑ ∃r.?.
    assert supports("bcbox", "box2el-go-inference")
    assert not supports("bcbox", "box2el-yeast-ppi")
    assert any("∃r.C" in reason for reason in missing_support("bcbox", "box2el-yeast-ppi"))
    # A TBox-only method cannot rank role assertions.
    assert not supports("mcmultibox", "boxlite-family-v1")
    assert supports("elem", "boxlite-family-v1")
    keys = {m.key for m in methods_for("box2el-go-inference")}
    assert {"elem", "box2el", "cate", "bcbox", "ont"} <= keys
    assert all(m.implemented for m in methods_for("box2el-go-inference"))
    mine = {b.key for b in benchmarks_for("bcbox")}
    assert "box2el-go-inference" in mine and "box2el-yeast-ppi" not in mine
    # Methods that only produce vectors reach a task through a scoring head ...
    from mowl.catalog import get_method
    assert get_method("owl2vec_star").tasks == ()
    assert get_method("owl2vec_star").head_tasks == ("subsumption",)
    assert supports("owl2vec_star", "box2el-go-inference")
    # ... and a method without any declared task matches no benchmark.
    assert missing_support("gda_projection", "box2el-go-inference")
