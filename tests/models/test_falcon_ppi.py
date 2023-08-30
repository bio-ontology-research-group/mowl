from unittest import TestCase

from tests.datasetFactory import PPIYeastSlimDataset
from mowl.models.falcon.examples.model_ppi import FALCON
from mowl.datasets import Dataset
from mowl.owlapi import OWLAPIAdapter
from java.util import HashSet
from java.io import File


class TestFalconPPI(TestCase):

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
        John = self.adapter.create_individual("http://John")
        Jane = self.adapter.create_individual("http://Jane")
        Robert = self.adapter.create_individual("http://Robert")
        Melissa = self.adapter.create_individual("http://Melissa")

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
        axioms.add(self.adapter.create_class_assertion(self.father, John))
        axioms.add(self.adapter.create_class_assertion(self.mother, Jane))
        axioms.add(self.adapter.create_class_assertion(self.male, Robert))
        axioms.add(self.adapter.create_class_assertion(self.female, Melissa))
        axioms.add(self.adapter.create_object_property_assertion(self.has_child, John, Robert))
        axioms.add(self.adapter.create_object_property_assertion(self.has_child, Jane, Robert))
        axioms.add(self.adapter.create_object_property_assertion(self.has_child, John, Melissa))
        axioms.add(self.adapter.create_object_property_assertion(self.has_child, Jane, Melissa))
        self.adapter.owl_manager.addAxioms(self.ontology, axioms)

        # self.ontology = self.adapter.owl_manager.loadOntologyFromOntologyDocument(File("deepgo/train_data.owl"))
        self.dataset = Dataset(self.ontology, validation=self.ontology)

    def test_ppi(self):
        """Test FALCON on PPI dataset. The test is not very strict, it just checks the \
correct syntax of the code."""
        model = FALCON(self.dataset, epochs=5, embed_dim=16)
        return_value = model.train()
        self.assertEqual(return_value, 1)
