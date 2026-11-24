"""End-to-end tests of the learnable al-cone model (mowl.models.alcone).

The discretised result of training is verified with the exact checker of
mowl.geometric.cones, which is independent of the losses.
"""
import numpy as np
import pytest
import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.geometric.cones import ALConeChecker, ALConeInterpretation
from mowl.models.alcone.model import ALConeModel
from mowl.nn.alc.alcone.module import ALConeModule
from mowl.nn.alc.module import ALCModule
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl/alcone#"


def _small_ontology():
    """A ⊑ B ⊔ C, B ⊓ C ⊑ ⊥, A ⊑ ∃r.¬A, ¬A ⊑ ∀r.A, A(a), ¬B(a), r(a, b)."""
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/alcone/small")
    A, B, C = (ad.create_class(NS + n) for n in "ABC")
    r = ad.create_object_property(NS + "r")
    a, b = ad.create_individual(NS + "a"), ad.create_individual(NS + "b")
    axs = HashSet()
    axs.add(ad.create_subclass_of(A, ad.create_object_union_of(B, C)))
    axs.add(ad.create_disjoint_classes(B, C))
    axs.add(ad.create_subclass_of(A, ad.create_object_some_values_from(
        r, ad.create_complement_of(A))))
    axs.add(ad.create_subclass_of(ad.create_complement_of(A),
                                  ad.create_object_all_values_from(r, A)))
    axs.add(ad.create_class_assertion(A, a))
    axs.add(ad.create_class_assertion(ad.create_complement_of(B), a))
    axs.add(ad.create_object_property_assertion(r, a, b))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


def _family_ontology():
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/alcone/family")
    c = {n: ad.create_class(NS + n) for n in ["Person", "Male", "Female", "Parent"]}
    r = ad.create_object_property(NS + "hasChild")
    axs = HashSet()
    axs.add(ad.create_subclass_of(c["Male"], c["Person"]))
    axs.add(ad.create_subclass_of(c["Female"], c["Person"]))
    axs.add(ad.create_disjoint_classes(c["Male"], c["Female"]))
    axs.add(ad.create_subclass_of(c["Person"],
                                  ad.create_object_union_of(c["Male"], c["Female"])))
    axs.add(ad.create_equivalent_classes(c["Parent"], ad.create_object_intersection_of(
        c["Person"], ad.create_object_some_values_from(r, c["Person"]))))
    axs.add(ad.create_subclass_of(c["Person"], ad.create_object_all_values_from(
        r, c["Person"])))
    axs.add(ad.create_class_assertion(c["Male"], ad.create_individual(NS + "john")))
    axs.add(ad.create_class_assertion(c["Female"], ad.create_individual(NS + "mary")))
    axs.add(ad.create_object_property_assertion(r, ad.create_individual(NS + "john"),
                                                ad.create_individual(NS + "mary")))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


@pytest.fixture(scope="module")
def trained():
    th.manual_seed(0)
    ds = _small_ontology()
    model = ALConeModel(ds, dim=4, margin=0.1, learning_rate=0.05, temperature=0.3)
    before = model.total_loss()
    model.train(epochs=400)
    return model, before


class TestALConeModel:

    def test_interface(self, trained):
        model, _ = trained
        assert isinstance(model.module, ALConeModule)
        assert isinstance(model.module, ALCModule)
        emb = model.class_embeddings
        assert all(v.shape == (8,) for v in emb.values())
        assert model.module.temperature == 0.0

    def test_loss_decreases(self, trained):
        model, before = trained
        assert model.total_loss() < before

    def test_trained_model_checked_by_exact_checker(self, trained):
        """Training reaches zero loss on this ontology, and the discretised al-cone
        interpretation is a model according to the exact checker."""
        model, _ = trained
        assert model.total_loss() == 0.0
        axioms, skipped = model.alc_axioms()
        assert skipped == []
        interp = model.discretise()
        assert isinstance(interp, ALConeInterpretation)
        assert ALConeChecker(interp).is_model(axioms)
        assert model.violations() == []
        for p in interp.individuals.values():
            assert np.any(p != 0)

    def test_scoring(self, trained):
        model, _ = trained
        s = model.score_subsumption(NS + "B", NS + "B")
        assert s == 0.0
        assert model.score_subsumption(NS + "A", NS + "C") <= 0.0


@pytest.mark.slow
def test_family_ontology_training():
    """Longer run on a small family ontology with disjunction, disjointness, an
    equivalence with ∃ and a ∀ axiom. Gradient training can stop in a local minimum (with
    the default settings 6 of 8 seeds reached zero loss in our runs); this seed does, and
    the exact checker confirms the discretised model."""
    th.manual_seed(4)
    model = ALConeModel(_family_ontology(), dim=8, learning_rate=0.05, temperature=0.3)
    before = model.total_loss()
    model.train(epochs=1000)
    after = model.total_loss()
    assert after < before
    assert after == 0.0
    assert model.violations() == []
