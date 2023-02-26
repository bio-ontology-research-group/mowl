from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.models import ELEmbeddings

from mowl.owlapi import OWLAPIAdapter
from copy import deepcopy

class TestSemanticModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
        
        adapter = OWLAPIAdapter()
        uncle = adapter.create_class("http://Uncle")
        has_sibling = adapter.create_object_property("http://hasSibling")
        father = adapter.create_class("http://Father")

        some_has_sibling = adapter.create_object_some_values_from(has_sibling, uncle)
        self.axiom = adapter.create_subclass_of(father, some_has_sibling)
        
        
    def test_reindex_embeddings(self):
        model = ELEmbeddings(self.dataset)
        
        class_embeddings_before = deepcopy(model.class_embeddings)
        property_embeddings_before = deepcopy(model.object_property_embeddings)

                
        model.add_axioms(self.axiom)
        class_embeddings_after = model.class_embeddings
        property_embeddings_after = model.object_property_embeddings

        self.assertIn("http://Uncle", class_embeddings_after)

        for cls, emb in class_embeddings_before.items():
            with self.subTest(cls=cls):
                self.assertEqual(emb.tolist(), class_embeddings_after[cls].tolist())

        self.assertIn("http://hasSibling", property_embeddings_after)
        for prop, emb in property_embeddings_before.items():
            with self.subTest(prop=prop):
                self.assertEqual(emb.tolist(), property_embeddings_after[prop].tolist())
