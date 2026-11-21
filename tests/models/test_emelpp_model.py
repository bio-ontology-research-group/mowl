from mowl import init_jvm
init_jvm("1g")

import os
import shutil
import tempfile
from unittest import TestCase

import torch as th
from jpype import java
from org.semanticweb.owlapi.model import IRI

from mowl.catalog.entries.emelpp import CARDS
from mowl.datasets import Dataset
from mowl.models.emelpp.model import EmELPP, EmELVar
from mowl.nn.el.emelpp.module import EmELPPModule, EmELVarModule
from mowl.owlapi import OWLAPIAdapter
from tests.datasetFactory import FamilyDataset


def ontology_with_role_axioms():
    """The four GCI normal forms, a role inclusion, a transitive property and a role chain."""
    adapter = OWLAPIAdapter()
    factory = adapter.data_factory
    classes = [factory.getOWLClass(IRI.create(f"http://test/C{i}")) for i in range(5)]
    props = [factory.getOWLObjectProperty(IRI.create(f"http://test/R{i}")) for i in range(3)]

    ont = adapter.owl_manager.createOntology(IRI.create("http://test/emelpp_role_axioms"))
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


class TestEmELPPModels(TestCase):

    @classmethod
    def setUpClass(cls):
        ont = ontology_with_role_axioms()
        cls.dataset = Dataset(ont, validation=ont, testing=ont)
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def make(self, cls, **kwargs):
        kwargs.setdefault("embed_dim", 8)
        kwargs.setdefault("batch_size", 8)
        kwargs.setdefault("model_filepath", os.path.join(self.tmp_dir, f"{cls.__name__}.pt"))
        return cls(self.dataset, **kwargs)

    def test_modules(self):
        self.assertIsInstance(self.make(EmELPP).module, EmELPPModule)
        self.assertNotIsInstance(self.make(EmELPP).module, EmELVarModule)
        self.assertIsInstance(self.make(EmELVar).module, EmELVarModule)

    def test_role_axioms_are_opt_in(self):
        model = self.make(EmELPP)
        self.assertFalse(model.load_role_axioms)
        self.assertNotIn("role_inclusion", model.training_datasets)
        self.assertNotIn("role_chain", model.training_datasets)

    def test_negative_sampling_corrupts_both_sides_of_gci2(self):
        """Section 3.3 of the paper: C ⊑ ∃R.D is corrupted in C or in D."""
        for cls in (EmELPP, EmELVar):
            config = self.make(cls).get_negative_sampling_config()
            self.assertEqual(config["gci2"], {"index_pool": "classes",
                                              "corrupt_column": [0, 2]})
            # only the normal forms with a negative loss are sampled
            self.assertEqual(set(config), {"gci2", "object_property_assertion"})

    def test_trains_with_role_axioms(self):
        for cls in (EmELPP, EmELVar):
            model = self.make(cls, load_role_axioms=True)
            self.assertIn("role_inclusion", model.training_datasets)
            self.assertIn("role_chain", model.training_datasets)
            self.assertEqual(len(model.training_datasets["role_inclusion"]), 1)
            self.assertEqual(len(model.training_datasets["role_chain"]), 2)
            before = [p.detach().clone() for p in model.module.parameters()]
            model.train(epochs=3, validate_every=10)
            after = list(model.module.parameters())
            for p in after:
                self.assertTrue(th.isfinite(p).all())
            # the role vectors received gradients from the role axiom losses
            self.assertFalse(th.equal(before[2], after[2].detach()))     # rel_embed

    def test_loss_decreases(self):
        """The summed loss of all normal forms (positive and negative) goes down."""
        th.manual_seed(0)
        model = self.make(EmELVar, load_role_axioms=True, learning_rate=0.01, margin=0.0)
        module = model.module

        def total():
            with th.no_grad():
                return sum(module(ds[:], name).sum().item()
                           for name, ds in model.training_datasets.items() if len(ds))

        start = total()
        model.train(epochs=150, validate_every=1000)
        self.assertLess(total(), start)

    def test_role_axiom_losses_decrease_when_trained(self):
        th.manual_seed(0)
        model = self.make(EmELPP, load_role_axioms=True, learning_rate=0.01, margin=0.0)
        module = model.module
        ri = model.training_datasets["role_inclusion"][:]
        rc = model.training_datasets["role_chain"][:]
        start = (module.role_inclusion_loss(ri).sum() + module.role_chain_loss(rc).sum()).item()
        model.train(epochs=200, validate_every=1000)
        end = (module.role_inclusion_loss(ri).sum() + module.role_chain_loss(rc).sum()).item()
        self.assertLess(end, start)

    def test_scoring_and_embeddings(self):
        model = self.make(EmELPP, load_role_axioms=True)
        model.train(epochs=2, validate_every=10)
        class_embeddings = model.class_embeddings
        self.assertEqual(len(class_embeddings), 7)
        self.assertEqual(len(next(iter(class_embeddings.values()))), 8)
        self.assertEqual(len(model.object_property_embeddings), 3)


class TestEmELPPOnFamily(TestCase):

    def test_trains_without_role_axioms(self):
        for cls in (EmELPP, EmELVar):
            model = cls(FamilyDataset(), embed_dim=10)
            model.train(epochs=2)
            self.assertTrue(all(th.isfinite(p).all() for p in model.module.parameters()))


class TestCards(TestCase):

    def test_cards_point_to_the_classes(self):
        by_key = {c.key: c for c in CARDS}
        self.assertEqual(set(by_key), {"emelpp", "emelvar"})
        self.assertIs(by_key["emelpp"].load(), EmELPP)
        self.assertIs(by_key["emelvar"].load(), EmELVar)
        for card in CARDS:
            self.assertEqual(card.region, "ball")
            self.assertFalse(card.intersection_closed)
