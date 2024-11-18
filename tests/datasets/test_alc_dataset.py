from inspect import classify_class_attrs
from unittest import TestCase

from tests.datasetFactory import FamilyDataset
from mowl.datasets import ALCDataset, Dataset
from mowl.owlapi.defaults import BOT, TOP
from mowl.owlapi.constants import R, THING
from mowl.owlapi.model import OWLClass, OWLAxiom
from mowl.owlapi.adapter import OWLAPIAdapter
from java.util import HashSet

from torch.utils.data import TensorDataset


class TestALCDataset(TestCase):

    @classmethod
    def setUpClass(self):
        self.adapter = OWLAPIAdapter()
        self.ontology = self.adapter.create_ontology("http://mowl/family")
        self.male = self.adapter.create_class("http://Male")
        self.female = self.adapter.create_class("http://Female")
        self.parent = self.adapter.create_class("http://Parent")
        self.person = self.adapter.create_class("http://Person")
        self.mother = self.adapter.create_class("http://Mother")
        self.father = self.adapter.create_class("http://Father")
        self.has_child = self.adapter.create_object_property("http://hasChild")
        axioms = HashSet()
        axioms.add(self.adapter.create_subclass_of(self.male, self.person))
        axioms.add(self.adapter.create_subclass_of(self.female, self.person))
        axioms.add(self.adapter.create_subclass_of(self.parent, self.person))
        axioms.add(self.adapter.create_subclass_of(self.mother, self.female))
        axioms.add(self.adapter.create_subclass_of(self.father, self.male))
        parent_and_male = self.adapter.create_object_intersection_of(self.parent, self.male)
        axioms.add(self.adapter.create_subclass_of(parent_and_male, self.father))
        parent_and_female = self.adapter.create_object_intersection_of(self.parent, self.female)
        axioms.add(self.adapter.create_subclass_of(parent_and_female, self.mother))
        male_or_female = self.adapter.create_object_union_of(self.male, self.female)
        axioms.add(self.adapter.create_equivalent_classes(male_or_female, self.person))
        not_male = self.adapter.create_complement_of(self.male)
        axioms.add(self.adapter.create_equivalent_classes(not_male, self.female))
        has_child_person = self.adapter.create_object_some_values_from(
            self.has_child, self.person)
        axioms.add(self.adapter.create_subclass_of(self.parent, has_child_person))
        self.adapter.owl_manager.addAxioms(self.ontology, axioms)
        self.dataset = Dataset(self.ontology, validation=self.ontology, testing=self.ontology)

    def test_param_types(self):
        """This should check if the parameters of ALCDataset are of the correct type"""

        with self.assertRaisesRegex(TypeError, "Parameter ontology must be of type \
org.semanticweb.owlapi.model.OWLOntology."):
            ALCDataset("ontology", self.dataset)

        with self.assertRaisesRegex(TypeError, "Optional parameter device must be of type str"):
            ALCDataset(self.ontology, self.dataset, device=1)

    def test_get_axiom_pattern(self):
        """This should check axiom patterns"""
        alc_dataset = ALCDataset(self.ontology, self.dataset)
        top = self.adapter.create_class(THING)
        r = self.adapter.create_object_property(R)

        subclass_of = self.adapter.create_subclass_of(self.parent, self.person)
        subclass_of_pat = self.adapter.create_subclass_of(top, top)

        self.assertEqual(
            subclass_of_pat, alc_dataset.get_axiom_pattern(subclass_of))

        has_child_person = self.adapter.create_object_some_values_from(
            self.has_child, self.person)
        subclass_of_object_some_values_from = self.adapter.create_subclass_of(
            self.parent, has_child_person)
        subclass_of_object_some_values_from_pat = self.adapter.create_subclass_of(
            top, self.adapter.create_object_some_values_from(r, top))

        self.assertEqual(
            subclass_of_object_some_values_from_pat,
            alc_dataset.get_axiom_pattern(subclass_of_object_some_values_from)
        )

    def test_get_grouped_axioms(self):
        """This should check grouped axiom patterns"""
        alc_dataset = ALCDataset(self.ontology, self.dataset)
        grouped_axioms = alc_dataset.get_grouped_axioms()
        self.assertIsInstance(grouped_axioms, dict)
        self.assertEqual(len(grouped_axioms), 5)

    def test_get_datasets(self):
        """This should check grouped axiom patterns"""
        alc_dataset = ALCDataset(self.ontology, self.dataset)
        grouped_datasets, rest_of_axioms = alc_dataset.get_datasets()
        self.assertIsInstance(grouped_datasets, dict)
        self.assertIsInstance(rest_of_axioms, list)
        for axiom, dataset in grouped_datasets.items():
            self.assertIsInstance(axiom, OWLAxiom)
            self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(grouped_datasets), 5)
