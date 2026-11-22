"""The descriptive training-design regimes of the LM reference codes (HiT, OnT) and the
recipes of Yang et al. (ISWC 2025)."""

import os

import pytest

from mowl.benchmarks.recipes import get_recipe
from mowl.benchmarks.runner import resolve_training_regime
from mowl.catalog.registry import list_regimes
from mowl.training.lm_reference import HiTCodeTraining, LMCodeTraining, OnTCodeTraining
from tests.benchmarks.tiny import scratch_dir, tiny_dataset

DESIGN_KEYS = ("hit-code-training", "ont-code-training")

_BENCHMARKS = [f"ont-{o}-{t}" for o in ("galen", "go", "anatomy")
               for t in ("prediction", "inference")]


def test_cards_registered_and_load():
    cards = {c.key: c for c in list_regimes()}
    for key in DESIGN_KEYS:
        assert key in cards, key
        cls = cards[key].load()
        assert cls.key == key
        assert issubclass(cls, LMCodeTraining)


def test_resolve_by_catalogue_key():
    assert isinstance(resolve_training_regime("ont-code-training"), OnTCodeTraining)
    assert isinstance(resolve_training_regime("hit-code-training"), HiTCodeTraining)


def test_attach_rejects_non_lm_models():
    class NotAModel:
        pass
    with pytest.raises(TypeError):
        OnTCodeTraining().attach(NotAModel())
    with pytest.raises(TypeError):
        HiTCodeTraining().attach(NotAModel())


@pytest.mark.parametrize("cls", (HiTCodeTraining, OnTCodeTraining))
def test_hit_takes_the_lm_designs_and_the_other_regimes(cls):
    from mowl.models.hit.model import HiT
    from mowl.nn.lm.encoders import BagOfWordsEncoder
    from mowl.training.margin import SlackMargin
    model = HiT(tiny_dataset(), encoder=BagOfWordsEncoder(dim=8),
                model_filepath=os.path.join(scratch_dir(), "hit.pt"))
    model.training_regime = cls()   # the design of the reference code: the combination the model names
    assert model.training_regime.key == cls.key
    assert model.training_regime.model is model
    model.training_regime = None
    assert model.training_regime is None
    # an EL hook regime is honoured now: the loop of HiT consults every hook
    model.training_regime = SlackMargin(0.1)
    assert model.training_regime.key == SlackMargin.key
    model.training_regime = None
    with pytest.raises(TypeError):
        model.training_regime = "ont-code-training"


@pytest.mark.parametrize("benchmark", _BENCHMARKS)
@pytest.mark.parametrize("method", ("ont", "hit"))
def test_recipes_registered(benchmark, method):
    recipe = get_recipe(benchmark, method)
    assert recipe.source.key == "yang2025ont"
    assert recipe.epochs == 1
    assert recipe.training_regime == "ont-code-training"
    assert resolve_training_regime(recipe.regime_spec()).key == "ont-code-training"
    kwargs = recipe.model_kwargs
    assert kwargs["verbalisation_style"] == "release"
    assert kwargs["clustering_margin"] == 3.0
    assert kwargs["centripetal_margin"] == 0.5
    assert kwargs["batch_size"] == 64
    assert kwargs["num_negatives"] == 1
    assert kwargs["learning_rate"] == 1e-5
    assert kwargs["warmup_steps"] == 500
    assert kwargs["centri_weight_grid"] == [i / 10 for i in range(20)]


def test_published_results_name_the_regime():
    from mowl.benchmarks.registry import published_results
    for benchmark in _BENCHMARKS:
        for method in ("ont", "hit"):
            pubs = published_results(benchmark=benchmark, method=method)
            assert pubs, (benchmark, method)
            for pub in pubs:
                assert pub.settings["training_regime"] == "ont-code-training"
