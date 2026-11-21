"""Concept2Box on an ontology: ontology view, instance view and cross-view links."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
from mowl.models import Concept2Box  # noqa: E402
from mowl.nn import Concept2BoxModule  # noqa: E402
from tests.models.kg_region_data import abox_dataset  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    th.manual_seed(0)
    return Concept2Box(abox_dataset(), embed_dim=8, learning_rate=0.02, batch_size=64)


def test_only_the_supported_forms_are_trained(model):
    assert isinstance(model.module, Concept2BoxModule)
    assert set(model.training_datasets) <= set(Concept2Box.SUPPORTED_GCIS)
    assert {"gci0", "gci2", "class_assertion", "object_property_assertion"} <= \
        set(model.training_datasets)


def test_negative_sampling_covers_the_four_forms(model):
    assert set(model.get_negative_sampling_config()) == set(Concept2Box.SUPPORTED_GCIS)


def test_training_reduces_the_loss_and_ranks_true_axioms_first(model):
    names = {n.rsplit("#", 1)[-1]: i for n, i in model.class_index_dict.items()}
    gci0 = model.training_datasets["gci0"][:]

    def mean_loss():
        with th.no_grad():
            return sum(model.module(ds[:], name).mean().item()
                       for name, ds in model.training_datasets.items())

    before = mean_loss()
    model.train(epochs=60)
    assert mean_loss() < before
    # scoring: a stated subclass axiom costs less than its reversal
    true_cost = model.module(gci0, "gci0").mean().item()
    reversed_cost = model.module(gci0[:, [1, 0]], "gci0").mean().item()
    assert true_cost < reversed_cost
    assert "A" in names


def test_scoring_interface(model):
    assert model.axiom_scoring == "module"
    assert set(model.scored_forms) == {"gci0", "gci2", "class_assertion",
                                       "object_property_assertion"}
    from mowl.evaluation.model_scoring import resolve_model_scorer, scorable_forms
    scorer, name = resolve_model_scorer(model)
    assert name == "loss"
    rows = model.training_datasets["gci0"][:]
    assert scorer(model, "gci0", rows).shape == (len(rows),)
    assert set(scorable_forms(model)) >= {"gci0", "gci2"}


def test_embeddings_are_available(model):
    assert len(model.class_embeddings) == len(model.class_index_dict)
    assert len(model.individual_embeddings) == len(model.individual_index_dict) > 0
