from inspect import classify_class_attrs
from unittest import TestCase

from tests.datasetFactory import FamilyDataset
from mowl.datasets import ALCDataset
from mowl.owlapi.defaults import BOT, TOP
from mowl.owlapi.constants import R, THING
from mowl.owlapi.model import OWLClass
from mowl.owlapi.adapter import OWLAPIAdapter
from java.util import HashSet

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
        
    def test_param_types(self):
        """This should check if the parameters of ALCDataset are of the correct type"""

        with self.assertRaisesRegex(TypeError, "Parameter ontology must be of type \
org.semanticweb.owlapi.model.OWLOntology."):
            ALCDataset("ontology")

        with self.assertRaisesRegex(TypeError, "Optional parameter class_index_dict must be of \
type dict"):
            ALCDataset(self.ontology, class_index_dict="class_index_dict")

        with self.assertRaisesRegex(TypeError, "Optional parameter object_property_index_dict \
must be of type dict"):
            ALCDataset(self.ontology,
                      object_property_index_dict="object_property_index_dict")
        with self.assertRaisesRegex(TypeError, "Optional parameter device must be of type str"):
            ALCDataset(self.ontology, device=1)

    def test_get_axiom_pattern(self):
        """This should check axiom patterns"""

        dataset = ALCDataset(self.ontology)
        top = self.adapter.create_class(THING)
        r = self.adapter.create_object_property(R)
        
        subclass_of = self.adapter.create_subclass_of(self.parent, self.person)
        subclass_of_pat = self.adapter.create_subclass_of(top, top)
        
        self.assertEqual(
            subclass_of_pat, dataset.get_axiom_pattern(subclass_of))

        has_child_person = self.adapter.create_object_some_values_from(
            self.has_child, self.person)
        subclass_of_object_some_values_from = self.adapter.create_subclass_of(
            self.parent, has_child_person)
        subclass_of_object_some_values_from_pat = self.adapter.create_subclass_of(
            top, self.adapter.create_object_some_values_from(r, top))
        
        self.assertEqual(
            subclass_of_object_some_values_from_pat,
            dataset.get_axiom_pattern(subclass_of_object_some_values_from)
        )

    def test_get_grouped_axioms(self):
        """This should check grouped axiom patterns"""
        dataset = ALCDataset(self.ontology)
        grouped_axioms = dataset.get_grouped_axioms()
        self.assertEqual(len(grouped_axioms), 5)
        
    def test_get_datasets(self):
        """This should check grouped axiom patterns"""
        dataset = ALCDataset(self.ontology)
        grouped_datasets = dataset.get_datasets()
        self.assertEqual(len(grouped_datasets), 5)
    
