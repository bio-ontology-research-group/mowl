"""Tests for FaithEL (mowl.models.faithel)."""
import itertools

import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.geometric.faithfulness import entailed
from mowl.geometric.normal_forms import NF1
from mowl.models import FaithEL
from tests.models.test_canonical_mixture import IND, NS, _write_family_abox_dataset


@pytest.fixture
def model(tmp_path):
    return FaithEL(_write_family_abox_dataset(tmp_path), batch_size=8, learning_rate=0.01,
                   neg_sampling=False, seed=0)


def test_declares_scoring_and_forms(model):
    assert model.axiom_scoring == "method"
    assert {"gci0", "gci2", "class_assertion", "object_property_assertion"} <= set(
        model.scored_forms)


def test_tbox_costs_are_zero_exactly_for_entailed_subsumptions(model):
    """The fixed TBox geometry is strongly faithful: cost 0 iff the axiom is entailed."""
    names = [NS + n for n in ("Father", "Mother", "Parent", "Male", "Female", "Person")]
    rows = th.tensor([[model.class_index_dict[a], model.class_index_dict[b]]
                      for a, b in itertools.permutations(names, 2)])
    costs = model.axiom_costs("gci0", rows)
    n_entailed = 0
    for (a, b), cost in zip(itertools.permutations(names, 2), costs.tolist()):
        truth = entailed(model.tbox.canonical_model, NF1(a, b))
        assert (cost == 0.0) == truth, (a, b)
        n_entailed += truth
    assert n_entailed >= 5


def test_moving_parameters_start_at_centroids_and_stay_in_radius(model):
    m = model.module
    assert th.allclose(m.concepts.weight, m.concept_centroid)
    model.train(3)
    dist = (m.concepts.weight - m.concept_centroid).norm(dim=1)
    assert float(dist.max()) <= model.radius + 1e-5
    both = th.cat([m.role_head.weight, m.role_tail.weight], dim=1)
    assert float((both - m.role_centroid).norm(dim=1).max()) <= model.radius + 1e-5


def test_training_decreases_loss_and_separates_true_assertions(model):
    idx = model.individual_index_dict
    cls = model.class_index_dict
    true = th.tensor([[idx[IND + "bob"], cls[NS + "Father"]],
                      [idx[IND + "alice"], cls[NS + "Mother"]],
                      [idx[IND + "carl"], cls[NS + "Male"]]])
    false = th.tensor([[idx[IND + "bob"], cls[NS + "Female"]],
                       [idx[IND + "alice"], cls[NS + "Male"]],
                       [idx[IND + "carl"], cls[NS + "Mother"]]])
    gap0 = float(model.axiom_costs("class_assertion", false).mean()
                 - model.axiom_costs("class_assertion", true).mean())  # noqa: W503
    model.train(60)
    assert model.loss_history[-1] < model.loss_history[0]
    mean_false = model.axiom_costs("class_assertion", false).mean()
    mean_true = model.axiom_costs("class_assertion", true).mean()
    gap1 = float(mean_false - mean_true)
    assert gap1 > gap0 and gap1 > 0
    role = th.tensor([[idx[IND + "bob"], model.object_property_index_dict[NS + "hasChild"],
                       idx[IND + "carl"]],
                      [idx[IND + "bob"], model.object_property_index_dict[NS + "hasChild"],
                       idx[IND + "alice"]]])
    c = model.axiom_costs("object_property_assertion", role)
    assert c[0] < c[1]


def test_tbox_only_dataset_scores_the_tbox_and_big_ontologies_are_refused():
    from tests.datasetFactory import FamilyDataset
    m = FaithEL(FamilyDataset())
    assert m.module.individual_model.centres.num_embeddings == 0
    m.train(1)
    with pytest.raises(ValueError, match="max_elements"):
        FaithEL(FamilyDataset(), max_elements=3)
