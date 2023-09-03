from unittest import TestCase
from tests.datasetFactory import FamilyDataset

from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import TOP

class TestAddAxiomsToDataset(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
        
        adapter = OWLAPIAdapter()
        
        aunt = adapter.create_class("http://Aunt")
        nephew = adapter.create_class("http://Nephew")
        self.father = adapter.create_class("http://Father")
        grandpa = adapter.create_class("http://Grandfather")
        top = adapter.create_class(TOP)

        has_parent = adapter.create_object_property("http://hasParent")
        has_sibling = adapter.create_object_property("http://hasSibling")
        has_aunt = adapter.create_object_property("http://hasAunt")
        self.has_child = adapter.create_object_property("http://hasChild")
        
        
        some_has_sibling = adapter.create_object_some_values_from(has_sibling, aunt)
        self.axiom1 = adapter.create_subclass_of(self.father, some_has_sibling)

        some_has_parent = adapter.create_object_some_values_from(has_parent, grandpa)
        self.axiom2 = adapter.create_subclass_of(self.father, some_has_parent)

        some_has_aunt = adapter.create_object_some_values_from(has_aunt, aunt)
        self.axiom3 = adapter.create_subclass_of(nephew, some_has_aunt)

    def test_add_new_axioms(self):
        """This should check that the new axioms are added to the dataset"""
        axioms_before = len(self.dataset.ontology.getAxioms())
        classes_before = len(self.dataset.classes)
        self.dataset.add_axioms(self.axiom2)

        axioms_after = len(self.dataset.ontology.getAxioms())
        classes_after = len(self.dataset.classes)
        
        self.assertEqual(axioms_before, axioms_after-1)
        self.assertEqual(classes_before, classes_after-1)
        


    def test_reindex_dataset(self):
        """This should check that the dataset is reindexed"""

        father_index_before = self.dataset.class_to_id[self.father]
        has_child_index_before = self.dataset.object_property_to_id[self.has_child]

        self.dataset.add_axioms(self.axiom1, self.axiom3)
        father_index_after = self.dataset.class_to_id[self.father]
        has_child_index_after = self.dataset.object_property_to_id[self.has_child]
        
        self.assertEqual(father_index_before, father_index_after-1)
        self.assertEqual(has_child_index_before, has_child_index_after-1)
        

