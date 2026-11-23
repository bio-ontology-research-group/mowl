"""A run is compared with the published numbers of its own variant.

A source often reports several variants of one method on one benchmark (two losses, two
dimensions, an ablation). The three-way table must put next to a run the numbers of the
variant the run follows, not the first numbers registered.
"""
from mowl.benchmarks.collect import _run_settings, collect, published_for
from mowl.benchmarks.registry import published_results
from mowl.benchmarks.recipes import list_recipes


def _result(benchmark, method, source, name, value, metric, model_kwargs=None):
    return {"benchmark": benchmark, "method": method, "seed": 0,
            "implementation": {"name": "parts"},
            "recipe": {"source": source, "name": name, "model_kwargs": model_kwargs or {}},
            "settings": dict(model_kwargs or {}),
            "metrics": {"all": {metric: value}}}


def _published(rows, metric):
    return {r["recipe"]: r["published"] for r in rows if r["metric"] == metric}


def test_each_loss_of_dfalc_is_compared_with_its_own_column():
    bench = "dfalc-revision-family-mask20"
    rows = collect([_result(bench, "dfalc", "wu2022", loss, 0.9, "success_rate",
                            {"loss_type": loss}) for loss in ("hierarchical", "rule")])
    assert _published(rows, "success_rate") == {"wu2022:hierarchical": 1.0,
                                                "wu2022:rule": 0.931}


def test_a_named_published_result_is_not_used_for_another_recipe():
    pubs = published_results(benchmark="kg-inductive-fb15k237-v1", method="reshuffle")
    by_recipe = {p.recipe: p.metrics["hits@10"] for p in pubs}
    assert set(by_recipe) == {"reshuffle", "no-loop", "square"}
    for name, value in by_recipe.items():
        assert published_for(pubs, name, {}).metrics["hits@10"] == value
    assert published_for(pubs, "another-variant", {}) is None


def test_without_a_name_the_stated_settings_decide():
    pubs = published_results(benchmark="regd-anatomy-inference", method="regd")
    dims = {p.settings["embed_dim"] for p in pubs}
    assert len(dims) > 1
    for dim in dims:
        chosen = published_for(pubs, "", _run_settings([{"settings": {"embed_dim": dim}}]))
        assert chosen.settings["embed_dim"] == dim


def test_every_recipe_with_published_variants_finds_one_of_its_own():
    """No recipe of a multi-variant source may fall back on numbers that state settings
    the recipe contradicts."""
    import collections
    groups = collections.defaultdict(list)
    for p in published_results():
        for m in p.metrics:
            groups[(p.benchmark, p.method, p.source.key, p.subset, m)].append(p)
    checked = 0
    for (bench, method, source, _subset, _metric), pubs in groups.items():
        if len(pubs) < 2:
            continue
        for recipe in list_recipes(benchmark=bench, method=method):
            if recipe.source.key != source:
                continue
            settings = _run_settings([{"recipe": {
                "name": recipe.name, "model_kwargs": dict(recipe.model_kwargs),
                "regime_kwargs": dict(recipe.regime_kwargs or {}),
                "training_regime": recipe.training_regime}}])
            chosen = published_for(pubs, recipe.name or "", settings)
            if chosen is None:
                continue
            checked += 1
            if chosen.recipe is not None:
                assert chosen.recipe == recipe.name
    assert checked > 50
