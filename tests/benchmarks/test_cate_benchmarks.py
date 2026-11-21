"""The CatE benchmarks end to end: the corrected evaluation regimes (explicit query
lists, strict Hits@k, both-direction PPI ranking), the cate-code-training regime, the
interaction loss and the recipes."""

import os

import numpy as np
import pytest
import torch as th

import mowl  # noqa: F401  (the tests package starts the JVM)
from mowl.datasets import PathDataset
from mowl.models.cate.model import CatE
from mowl.owlapi import OWLAPIAdapter
from tests.benchmarks.tiny import NS, scratch_dir, tiny_dataset, write_tiny


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------

def test_strict_hits():
    from mowl.evaluation.ranking_metrics import summarise_ranks
    ranks = [1, 2, 3, 10, 11]
    usual = summarise_ranks(ranks, 20, hits=(1, 10))
    strict = summarise_ranks(ranks, 20, hits=(1, 10), strict_hits=True)
    assert usual["hits@1"] == pytest.approx(0.2)
    assert strict["hits@1"] == 0.0  # rank < 1 is impossible
    assert usual["hits@10"] == pytest.approx(0.8)
    assert strict["hits@10"] == pytest.approx(0.6)  # ranks 1, 2, 3 only
    assert strict["mr"] == usual["mr"] and strict["mrr"] == usual["mrr"]


# ---------------------------------------------------------------------------------------
# Evaluation regimes
# ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_cate():
    ds = tiny_dataset()
    th.manual_seed(0)
    model = CatE(ds, embed_dim=8, batch_size=64, learning_rate=0.05,
                 existentials={NS + "r": "evaluation_classes"},
                 model_filepath=os.path.join(scratch_dir(), "cate_bench.pt"))
    model.train(epochs=20)
    return model, ds


def test_explicit_queries_and_exclude_head(trained_cate):
    """The released query lists (dataset attribute) are ranked instead of the test
    axioms, and a query's head is removed from its own candidates."""
    from mowl.evaluation.filtered_ranking import cate_ranking
    model, ds = trained_cate
    ds.cate_test_query_pairs = [(NS + "A", NS + "C"), (NS + "E", NS + "C")]
    regime = cate_ranking("gci0", candidates="classes", closure=False,
                          queries="dataset:cate_test_query_pairs", exclude_head=True)
    res = regime.evaluate(model, ds)["all"]
    assert regime.info["queries"] == 2
    assert 1 <= res["f_mr"] <= regime.info["num_candidates"]
    # Hits@k is strict: a rank-1 query counts at Hits@10 but never at Hits@1.
    assert res["f_hits@100"] >= res["f_hits@10"] >= 0.0
    del ds.cate_test_query_pairs


def test_unknown_queries_are_dropped(trained_cate):
    from mowl.evaluation.filtered_ranking import cate_ranking
    model, ds = trained_cate
    ds.cate_test_query_pairs = [(NS + "A", NS + "C"), ("http://unknown", NS + "C")]
    regime = cate_ranking("gci0", candidates="classes", closure=False,
                          queries="dataset:cate_test_query_pairs", exclude_head=True)
    regime.evaluate(model, ds)["all"]
    assert regime.info["queries"] == 1 and regime.info["skipped_axioms"] == 1
    del ds.cate_test_query_pairs


def test_closure_filtering_is_stronger_than_train(trained_cate):
    from mowl.evaluation.filtered_ranking import cate_ranking
    model, ds = trained_cate
    train_only = cate_ranking("gci0", candidates="classes", closure=False)
    closure = cate_ranking("gci0", candidates="classes", closure=True)
    r1 = train_only.evaluate(model, ds)["all"]
    r2 = closure.evaluate(model, ds)["all"]
    assert r2["f_mr"] <= r1["f_mr"]
    assert closure.info["queries"] == train_only.info["queries"]


def test_ppi_ranking_both_directions(trained_cate):
    """Every test interaction is ranked in both directions and the metrics are averaged
    over the 2 x n ranks."""
    from mowl.evaluation.filtered_ranking import cate_ppi_ranking
    model, ds = trained_cate
    regime = cate_ppi_ranking(NS + "r")
    res = regime.evaluate(model, ds)["all"]
    n_inter = regime.info["queries"]
    assert n_inter >= 1 and regime.info["directions"] == 2
    assert regime.info["num_candidates"] == 3  # P1, P2, P3
    assert set(res) == {"f_mr", "f_mrr", "f_hits@1", "f_hits@3", "f_hits@10",
                        "f_hits@100", "f_auc"}
    assert 1 <= res["f_mr"] <= regime.info["num_candidates"]
    # Hits@k uses rank <= k here (the reference PPI evaluator), so a rank-1 direction
    # counts at Hits@1.
    assert res["f_hits@1"] >= res["f_hits@10"] or res["f_hits@1"] == 0.0


# ---------------------------------------------------------------------------------------
# Training regime and interaction loss
# ---------------------------------------------------------------------------------------

class _TinyPPI(PathDataset):
    """The tiny dataset seen as a PPI dataset: all three proteins are evaluation
    classes and the evaluation property is r."""

    def __init__(self, *paths):
        super().__init__(*paths)
        adapter = OWLAPIAdapter()
        proteins = {adapter.create_class(NS + name) for name in ("P1", "P2", "P3")}
        from mowl.datasets.base import OWLClasses
        self._proteins = OWLClasses(proteins)

    def get_evaluation_property(self):
        return NS + "r"

    @property
    def evaluation_classes(self):
        return self._proteins, self._proteins


def test_code_training_regime_and_interaction_loss():
    from mowl.training import CateCodeTraining
    train, valid, test = write_tiny(scratch_dir())
    ds = _TinyPPI(train, valid, test)
    th.manual_seed(0)
    model = CatE(ds, embed_dim=8, batch_size=16, learning_rate=0.05,
                 model_filepath=os.path.join(scratch_dir(), "cate_ppi.pt"))
    assert model.negative_aggregation == "per_negative" and model.filter_negatives
    model.training_regime = CateCodeTraining(interaction_loss=True)
    assert model.negative_aggregation == "mean" and not model.filter_negatives
    assert model.interaction_loss
    history = model.train(epochs=5)
    assert len(history) == 5 and all(np.isfinite(history))
    heads, ex = model._interaction_ids()
    expected = {model.lattice.class_to_id[NS + p] for p in ("P1", "P2", "P3")}
    assert set(heads.tolist()) == expected
    assert len(ex) == 3  # the elements ∃r.P1, ∃r.P2, ∃r.P3


def test_code_training_via_runner():
    """The runner resolves the regime by key; the PPI recipe uses it with the
    interaction loss."""
    from mowl.benchmarks.runner import run_benchmark
    train, valid, test = write_tiny(scratch_dir())
    ds = _TinyPPI(train, valid, test)
    res = run_benchmark("cate-yeast-ppi", method="cate", dataset=ds, epochs=2,
                        model_kwargs={"embed_dim": 8, "batch_size": 64,
                                      "learning_rate": 0.05,
                                      "existentials": {NS + "r": "evaluation_classes"}},
                        protocol_kwargs={"relation": NS + "r"},
                        training_regime=("cate-code-training",
                                         {"interaction_loss": True}))
    assert res.training_regime == "cate-code-training"
    assert res.metrics["all"]["f_mr"] >= 1


def test_code_training_needs_cate():
    from mowl.models.elbe.model import ELBE
    from mowl.training import CateCodeTraining
    with pytest.raises(TypeError):
        CateCodeTraining().attach(ELBE(tiny_dataset(), embed_dim=2))


# ---------------------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------------------

def test_cate_recipes_resolve():
    from mowl.benchmarks.recipes import get_recipe, list_recipes
    from mowl.benchmarks.entries._refs import ZHAPA2023
    recipes = list_recipes(source=ZHAPA2023.key)
    assert len(recipes) == 4
    cases = (("cate-ore1-membership", 50000, 200), ("cate-go-completion", 10000, 200),
             ("cate-foodon-completion", 10000, 200), ("cate-yeast-ppi", 10000, 256))
    for benchmark, epochs, dim in cases:
        recipe = get_recipe(benchmark, "cate")
        assert recipe.epochs == epochs and recipe.model_kwargs["embed_dim"] == dim
        assert recipe.training_regime == "cate-code-training"
        assert recipe.validate_every == 100
    ppi = get_recipe("cate-yeast-ppi", "cate")
    assert ppi.model_kwargs["saturation_steps"] == 1
    assert ppi.regime_kwargs["interaction_loss"] is True
    # The completion recipes match the released training graphs (reference projector,
    # trivial morphisms removed); the PPI graph keeps them.
    for benchmark in ("cate-ore1-membership", "cate-go-completion",
                      "cate-foodon-completion"):
        recipe = get_recipe(benchmark, "cate")
        assert recipe.model_kwargs["complete_negation"] is False
        assert recipe.model_kwargs["obvious_morphisms"] is False
    assert ppi.model_kwargs["complete_negation"] is False
    assert "obvious_morphisms" not in ppi.model_kwargs
    assert get_recipe("cate-ore1-membership", "cate").model_kwargs["p"] == 1


@pytest.mark.slow
def test_cate_subsumption_datasets():
    """The completion datasets expose the released cleaned query lists (needs the
    ~100 MB Zenodo archive; excluded from CI)."""
    from mowl.datasets.builtin.zhapa2023 import (CatEFoodOnSubsumptionDataset,
                                                 CatEGOSubsumptionDataset)
    go = CatEGOSubsumptionDataset()
    foodon = CatEFoodOnSubsumptionDataset()
    assert len(go.cate_test_query_pairs) == 11893
    assert len(foodon.cate_test_query_pairs) == 3632
    assert len(go.cate_validation_query_pairs) == 5935
    assert len(foodon.cate_validation_query_pairs) == 1888
    classes = {str(c) for c in go.classes.as_str}
    assert all(h in classes and t in classes for h, t in go.cate_test_query_pairs[:50])
