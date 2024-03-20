from unittest import TestCase
from tests.datasetFactory import FamilyDataset, PPIYeastSlimDataset
from mowl.models.elembeddings.examples.model_ppi import ELEmPPI
from mowl.models import ELEmbeddings

from mowl.owlapi import OWLAPIAdapter
from copy import deepcopy
import mowl.error.messages as msg
import os

class TestSemanticModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
        self.ppi_dataset = PPIYeastSlimDataset()
        
        adapter = OWLAPIAdapter()
        aunt = adapter.create_class("http://Aunt")
        has_sibling = adapter.create_object_property("http://hasSibling")
        father = adapter.create_class("http://Father")

        some_has_sibling = adapter.create_object_some_values_from(has_sibling, aunt)
        self.axiom = adapter.create_subclass_of(father, some_has_sibling)
        
        
    def test_reindex_embeddings(self):
        model = ELEmbeddings(self.dataset)
        
        class_embeddings_before = deepcopy(model.class_embeddings)
        property_embeddings_before = deepcopy(model.object_property_embeddings)

                
        model.add_axioms(self.axiom)
        class_embeddings_after = model.class_embeddings
        property_embeddings_after = model.object_property_embeddings

        self.assertIn("http://Aunt", class_embeddings_after)

        for cls, emb in class_embeddings_before.items():
            with self.subTest(cls=cls):
                self.assertEqual(emb.tolist(), class_embeddings_after[cls].tolist())

        self.assertIn("http://hasSibling", property_embeddings_after)
        for prop, emb in property_embeddings_before.items():
            with self.subTest(prop=prop):
                self.assertEqual(emb.tolist(), property_embeddings_after[prop].tolist())


    def test_from_pretrained(self):
        model = ELEmbeddings(self.dataset)

        with self.assertRaisesRegex(TypeError, "Parameter model must be a string pointing to the model file."):
            model.from_pretrained(1)

        with self.assertRaisesRegex(FileNotFoundError, "Pretrained model path does not exist"):
            model.from_pretrained("path")

        
            
    def test_train_after_pretrained(self):
        first_model = ELEmPPI(self.ppi_dataset, model_filepath="first_semantic_model", epochs=3)
        first_model.train(validate_every=1)

        first_semantic_model = first_model.model_filepath
        
        self.assertTrue(os.path.exists(first_semantic_model))

        second_model = ELEmPPI(self.ppi_dataset, epochs=2)
        second_model.from_pretrained(first_semantic_model)

        self.assertNotEqual(second_model.model_filepath, first_semantic_model)
        second_model.train()



