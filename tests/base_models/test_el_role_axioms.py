"""Tests for ``EmbeddingELModel(load_role_axioms=...)``: the EL++ role axioms are opt-in,
and asking for them with a module that cannot consume them fails loudly and early."""

import mowl
from mowl import init_jvm
init_jvm("1g")

import os
import shutil
import tempfile
from unittest import TestCase

import torch as th
from org.semanticweb.owlapi.model import IRI
from jpype import java

from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import Dataset
from mowl.base_models import EmbeddingELModel
from mowl.models import ELEmbeddings
from mowl.nn import ELEmModule

ROLE_AXIOM_GCIS = ("role_inclusion", "role_chain")


def ontology_with_role_axioms():
    """Ontology with the four GCI normal forms, a role inclusion, a transitive property and
    a role chain."""
    adapter = OWLAPIAdapter()
    factory = adapter.data_factory

    classes = [factory.getOWLClass(IRI.create(f"http://test/C{i}")) for i in range(5)]
    props = [factory.getOWLObjectProperty(IRI.create(f"http://test/R{i}")) for i in range(3)]

    ont = adapter.owl_manager.createOntology(IRI.create("http://test/role_axiom_model"))
    ont.addAxiom(factory.getOWLSubClassOfAxiom(classes[0], classes[1]))
    ont.addAxiom(factory.getOWLSubClassOfAxiom(
        factory.getOWLObjectIntersectionOf(classes[1], classes[2]), classes[3]))
    ont.addAxiom(factory.getOWLSubClassOfAxiom(
        classes[1], factory.getOWLObjectSomeValuesFrom(props[0], classes[2])))
    ont.addAxiom(factory.getOWLSubClassOfAxiom(
        factory.getOWLObjectSomeValuesFrom(props[0], classes[3]), classes[4]))
    ont.addAxiom(factory.getOWLSubObjectPropertyOfAxiom(props[0], props[1]))
    ont.addAxiom(factory.getOWLTransitiveObjectPropertyAxiom(props[1]))
    chain = java.util.ArrayList()
    chain.add(props[1])
    chain.add(props[2])
    ont.addAxiom(factory.getOWLSubPropertyChainOfAxiom(chain, props[0]))
    return ont


class RoleCapableModule(ELEmModule):
    """Minimal module that implements the role axiom normal forms."""

    role_axiom_capable = True

    def role_inclusion_loss(self, gci, neg=False):
        return th.zeros(gci.shape[0], 1, requires_grad=True)

    def role_chain_loss(self, gci, neg=False):
        return th.zeros(gci.shape[0], 1, requires_grad=True)


class RoleCapableModel(EmbeddingELModel):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("load_role_axioms", True)
        super().__init__(*args, **kwargs)
        self.init_module()

    def init_module(self):
        self.module = RoleCapableModule(len(self.class_index_dict),
                                        len(self.object_property_index_dict),
                                        len(self.individual_index_dict),
                                        embed_dim=self.embed_dim).to(self.device)


class RoleIncapableModel(EmbeddingELModel):
    """A model that asks for the role axioms but whose module cannot train on them."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("load_role_axioms", True)
        super().__init__(*args, **kwargs)
        self.init_module()

    def init_module(self):
        self.module = ELEmModule(len(self.class_index_dict),
                                 len(self.object_property_index_dict),
                                 len(self.individual_index_dict),
                                 embed_dim=self.embed_dim).to(self.device)


class TestRoleAxiomLoading(TestCase):

    @classmethod
    def setUpClass(cls):
        ont = ontology_with_role_axioms()
        cls.dataset = Dataset(ont, validation=ont, testing=ont)
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def model_filepath(self, name):
        return os.path.join(self.tmp_dir, name)

    def test_role_axioms_are_not_loaded_by_default(self):
        """The role axioms of the ontology are invisible to a model that did not ask for
        them, in every split and in the dataloaders."""
        model = ELEmbeddings(self.dataset, embed_dim=8, batch_size=4,
                             model_filepath=self.model_filepath("elem.pt"))

        self.assertFalse(model.load_role_axioms)

        for datasets in (model.training_datasets, model.validation_datasets,
                         model.testing_datasets):
            for gci_name in ROLE_AXIOM_GCIS:
                self.assertNotIn(gci_name, datasets)

        for gci_name in ROLE_AXIOM_GCIS:
            self.assertNotIn(gci_name, model.training_dataloaders)

    def test_default_model_still_trains(self):
        """The role axioms of the ontology must not break a model that ignores them."""
        model = ELEmbeddings(self.dataset, embed_dim=8, batch_size=4,
                             model_filepath=self.model_filepath("elem_train.pt"))
        model.eval_gci_name = "gci0"
        model.train(epochs=1)

    def test_role_axioms_are_loaded_when_requested(self):
        model = RoleCapableModel(self.dataset, embed_dim=8, batch_size=4,
                                 model_filepath=self.model_filepath("capable.pt"))

        for gci_name in ROLE_AXIOM_GCIS:
            self.assertIn(gci_name, model.training_datasets)
            self.assertIn(gci_name, model.training_dataloaders)

        self.assertEqual(model.training_datasets["role_inclusion"].data.shape[1], 2)
        self.assertEqual(model.training_datasets["role_chain"].data.shape[1], 3)

    def test_capable_module_trains_on_role_axioms(self):
        model = RoleCapableModel(self.dataset, embed_dim=8, batch_size=4,
                                 model_filepath=self.model_filepath("capable_train.pt"))
        model.eval_gci_name = "gci0"
        model.train(epochs=1)

    def test_incapable_module_is_rejected_before_training(self):
        """Loading the role axioms without a module that implements their losses is a
        configuration error, reported at the start of training rather than mid-epoch."""
        model = RoleIncapableModel(self.dataset, embed_dim=8, batch_size=4,
                                   model_filepath=self.model_filepath("incapable.pt"))
        model.eval_gci_name = "gci0"

        for gci_name in ROLE_AXIOM_GCIS:
            self.assertIn(gci_name, model.training_datasets)

        with self.assertRaises(NotImplementedError) as ctx:
            model.train(epochs=1)

        message = str(ctx.exception)
        self.assertIn("role_axiom_capable", message)
        self.assertIn("load_role_axioms", message)

    def test_type_check(self):
        with self.assertRaisesRegex(TypeError, "load_role_axioms must be of type bool"):
            RoleCapableModel(self.dataset, embed_dim=8, batch_size=4,
                             model_filepath=self.model_filepath("bad.pt"),
                             load_role_axioms="yes")
