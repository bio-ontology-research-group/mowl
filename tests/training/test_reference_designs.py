"""Margin regimes, reference-code training designs, the TransBox ranking variant and
recipes."""
import os

import pytest
import torch as th

from mowl.datasets.builtin import FamilyDataset
from mowl.models import ELBE, BoxSquaredEL, ELEmbeddings, TransBox
from mowl.training import (SlackMargin, StrictMargin, Box2ELCodeTraining,
                           TransBoxCodeTraining, margin_semantics)
from mowl.evaluation.el_scoring import (centre_distance_scores,
                                        centre_distance_zero_empty_scores,
                                        empty_intersection, resolve_scorer)
from tests.benchmarks.tiny import scratch_dir, tiny_dataset


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


def _model(cls, dataset, **kwargs):
    return cls(dataset, embed_dim=3,
               model_filepath=os.path.join(scratch_dir(), f"{cls.__name__}.pt"), **kwargs)


@pytest.mark.parametrize("cls,semantics", [(ELBE, "strict"), (TransBox, "strict"),
                                           (BoxSquaredEL, "slack"),
                                           (ELEmbeddings, "slack")])
def test_margin_is_set_by_meaning(dataset, cls, semantics):
    model = _model(cls, dataset)
    assert margin_semantics(model) == semantics
    name = model.module.margin_attribute
    model.training_regime = SlackMargin(0.2)
    assert getattr(model.module, name) == (0.2 if semantics == "slack" else -0.2)
    model.training_regime = StrictMargin(0.2)
    assert getattr(model.module, name) == (0.2 if semantics == "strict" else -0.2)


@pytest.mark.parametrize("cls", [ELBE, TransBox, BoxSquaredEL])
def test_slack_and_strict_inclusion_loss(dataset, cls):
    """A box that sticks out of its superclass box by 0.1 has zero loss under a slack
    margin of 0.2 and positive loss under a strict margin; a box nested with a gap of
    0.3 has zero loss under both."""
    model = _model(cls, dataset)
    module = model.module
    tables = {ELBE: ("class_embed", "class_offset"),
              TransBox: ("class_center_embedding", "class_offset_embedding"),
              BoxSquaredEL: ("class_center", "class_offset")}[cls]
    centre, offset = (getattr(module, t) for t in tables)
    with th.no_grad():
        centre.weight[0] = 0.0
        centre.weight[1] = 0.0
        centre.weight[2] = 0.0
        offset.weight[0] = 1.1   # C sticks out of D by 0.1
        offset.weight[1] = 1.0   # D
        offset.weight[2] = 0.7   # E is inside D with a gap of 0.3
    sticks_out = th.tensor([[0, 1]])
    nested = th.tensor([[2, 1]])
    model.training_regime = SlackMargin(0.2)
    assert float(module(sticks_out, "gci0").sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(module(nested, "gci0").sum()) == pytest.approx(0.0, abs=1e-6)
    model.training_regime = StrictMargin(0.2)
    assert float(module(sticks_out, "gci0").sum()) > 0
    assert float(module(nested, "gci0").sum()) == pytest.approx(0.0, abs=1e-6)


def test_margin_regime_rejects_negative_and_undeclared(dataset):
    with pytest.raises(ValueError):
        SlackMargin(-0.1)
    model = _model(ELBE, dataset)
    model.module.margin_semantics = None
    with pytest.raises(TypeError):
        model.training_regime = SlackMargin(0.1)


def test_box2el_code_training_composition(dataset):
    regime = Box2ELCodeTraining(batch_size=4, num_negatives=2, margin=0.1, neg_target=2.0,
                                lr_milestones=[2], validate_every=2)
    assert regime.key == "box2el-code-training"
    assert [r.key for r in regime.regimes] == [
        "per-normal-form-minibatch", "per-column-corruption", "slack-inclusion-margin",
        "mse-to-target", "step-lr-decay", "validation-rank-selection"]
    model = _model(ELBE, tiny_dataset())
    model.training_regime = regime
    assert model.module.margin == -0.1
    model.train(epochs=4)
    assert regime.regimes[-1].best_epoch is not None
    assert TransBoxCodeTraining().regimes[0].batch_size == 1024
    assert TransBoxCodeTraining().key == "transbox-code-training"


def test_zero_score_on_empty_intersection(dataset):
    model = _model(ELBE, dataset)
    module = model.module
    with th.no_grad():
        module.class_embed.weight[0] = 0.0
        module.class_embed.weight[1] = 5.0   # disjoint from class 0
        module.class_embed.weight[2] = 0.2   # overlaps class 0
        module.class_offset.weight[:3] = 1.0
    rows = th.tensor([[0, 1, 3], [0, 2, 3]])
    assert empty_intersection(model, rows).tolist() == [True, False]
    plain = centre_distance_scores(model, "gci1", rows)
    zeroed = centre_distance_zero_empty_scores(model, "gci1", rows)
    assert float(zeroed[0]) == 0.0 and float(plain[0]) > 0
    assert float(zeroed[1]) == pytest.approx(float(plain[1]))
    # Other normal forms are not affected.
    assert th.equal(centre_distance_zero_empty_scores(model, "gci0", rows[:, :2]),
                    centre_distance_scores(model, "gci0", rows[:, :2]))
    assert resolve_scorer(model, "centre_distance_zero_empty")[1] \
        == "centre_distance_zero_empty"


def test_recipes_and_benchmarks_are_consistent():
    from mowl.benchmarks import get_benchmark, list_recipes, get_recipe, published_results
    from mowl.catalog import get_regime, get_method
    recipes = list_recipes()
    assert recipes
    for r in recipes:
        get_benchmark(r.benchmark)
        get_method(r.method)
        assert get_regime(r.training_regime).implemented
    r = get_recipe("box2el-galen-inference", "box2el")
    assert r.regime_kwargs["lr_milestones"] == [2000]
    assert get_recipe("transbox-go-prediction", "transbox").epochs == 5000
    with pytest.raises(KeyError):
        get_recipe("box2el-go-prediction", "transbox")
    # The TransBox paper's numbers are filed under its own evaluation regime.
    assert not published_results(benchmark="box2el-go-prediction", method="transbox")
    assert published_results(benchmark="transbox-go-prediction", method="transbox")
    assert get_benchmark("transbox-go-prediction").protocol.endswith(
        "TransBoxPredictionProtocol")
