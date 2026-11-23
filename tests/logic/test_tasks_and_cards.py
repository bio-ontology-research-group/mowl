"""Inference tasks, and the expressivity and tasks declared on catalogue cards."""

import importlib

from mowl.catalog import list_methods, list_regimes
from mowl.logic import (TASKS, COMPLETION_TASKS, Construct, LogicalStatus, Target, get_task,
                        get_completion_task)


def test_tasks_are_consistent():
    assert get_task("instance-checking").as_subsumption == "{a} ⊑ C"
    for task in TASKS.values():
        for name in task.needs:
            assert hasattr(Construct, name), (task.key, name)
    for task in COMPLETION_TASKS.values():
        assert task.inference_task in TASKS, task.key
    assert get_completion_task("link-prediction").inference_task == "role-assertion"
    assert {s.value for s in LogicalStatus} == {"entailed", "contradictory", "undetermined"}
    assert len(Target) == 3


#: Implemented methods that are not defined on a description logic (see their ``logic``).
#: graph_kge states what its widest projectors read (its default factory uses DL2Vec).
NO_DL = {"gda_projection"}


def test_every_implemented_method_states_its_expressivity():
    for card in list_methods(implemented=True):
        if card.key in NO_DL:
            assert card.supported_expressivity is None
            continue
        assert card.supported_expressivity is not None, card.key


def test_needed_constructs_of_declared_tasks_are_supported():
    """A method that declares an ABox task must support assertions, and one that declares
    satisfiability or disjointness must support ⊥."""
    for card in list_methods(implemented=True):
        logic = card.supported_expressivity
        if logic is None:
            assert not card.tasks, card.key
            continue
        for key in card.tasks:
            task = TASKS[key]
            if task.box.value == "ABox":
                assert logic.has_abox, (card.key, key)
            if "BOTTOM" in task.needs and card.key != "falcon":
                assert Construct.BOTTOM in logic, (card.key, key)


#: EL methods whose module does not show its expressivity through overridden losses: the
#: bottom and assertion losses of mcmultibox raise, and canonical_mixture takes its
#: semantics from the canonical model.
NOT_MECHANICAL = {"mcmultibox", "canonical_mixture"}


def test_el_cards_agree_with_their_modules():
    """The ABox, ⊥ and role axiom support stated on a card is what the module implements."""
    from mowl.nn import ELModule
    from mowl.representation.geometry import ComposedGeometry
    from tests.nn.el_module_zoo import all_el_module_classes, make
    zoo = set(all_el_module_classes())

    def overrides(cls, name):
        return getattr(cls, name) is not getattr(ELModule, name)

    checked = 0
    for card in list_methods(implemented=True):
        if not card.module or card.key in NOT_MECHANICAL:
            continue
        module_name, class_name = card.module.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        if not issubclass(cls, ELModule):
            continue
        logic = card.supported_expressivity
        if issubclass(cls, ComposedGeometry):
            # built from parts: every loss is generated, so the instance says what it reads
            if cls not in zoo:
                continue
            forms = make(cls).supported_forms
            code = ({"class_assertion", "object_property_assertion"} <= forms,
                    {"gci0_bot", "gci1_bot"} <= forms, "role_chain" in forms)
        else:
            code = (overrides(cls, "class_assertion_loss")
                    and overrides(cls, "object_property_assertion_loss"),
                    overrides(cls, "gci0_bot_loss") and overrides(cls, "gci1_bot_loss"),
                    bool(getattr(cls, "role_axiom_capable", False)))
        stated = (logic.has_abox, Construct.BOTTOM in logic, Construct.ROLE_CHAIN in logic)
        assert stated == code, (card.key, card.expressivity, code)
        checked += 1
    assert checked >= 10


def test_evaluation_regimes_name_known_tasks():
    with_task = [c for c in list_regimes(kind="evaluation") if c.task]
    assert len(with_task) >= 20
    for card in with_task:
        assert card.task in TASKS or card.task in COMPLETION_TASKS


def test_model_reports_axioms_beyond_its_expressivity():
    import pytest
    from mowl.datasets import PathDataset
    from mowl.models import ELEmbeddings
    from mowl.owlapi import OWLAPIAdapter
    from org.semanticweb.owlapi.model import IRI
    import os
    import tempfile

    ad = OWLAPIAdapter()
    ns = "http://mowl.borg/model_expressivity#"
    ont = ad.create_ontology("http://mowl.borg/model_expressivity")
    a, b = ad.create_class(ns + "A"), ad.create_class(ns + "B")
    r = ad.create_object_property(ns + "r")
    for axiom in (ad.create_subclass_of(a, b),
                  ad.create_subclass_of(a, ad.create_object_some_values_from(r, b)),
                  ad.create_subclass_of(b, ad.create_object_all_values_from(r, a))):
        ad.owl_manager.addAxiom(ont, axiom)
    path = os.path.join(tempfile.mkdtemp(prefix="mowl_expressivity_"), "ontology.owl")
    ad.owl_manager.saveOntology(ont, IRI.create("file:" + path))

    with pytest.warns(UserWarning, match="1 of 3 logical axioms go beyond EL⊥ with ABox"):
        model = ELEmbeddings(PathDataset(path), embed_dim=4)
    assert model.card.key == "elem"
    assert str(model.expressivity) == "EL⊥ with ABox"
    assert model.expressivity_report.by_axiom_type == {"SubClassOf": 1}
