from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.models import SyntacticPlusW2VModel
from mowl.owlapi import OWLAPIAdapter
from copy import deepcopy
import mowl.error.messages as msg
import os

class TestSyntacticModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
                
        adapter = OWLAPIAdapter()
        aunt = adapter.create_class("http://Aunt")
        has_sibling = adapter.create_object_property("http://hasSibling")
        father = adapter.create_class("http://Father")

        some_has_sibling = adapter.create_object_some_values_from(has_sibling, aunt)
        self.axiom = adapter.create_subclass_of(father, some_has_sibling)
        
    def test_reindex_embeddings(self):
        model = SyntacticPlusW2VModel(self.dataset)
        model.set_w2v_model(min_count=1)
        model.generate_corpus(save=True, with_annotations=True)
        model.train()
        class_embeddings_before = deepcopy(model.class_embeddings)
        property_embeddings_before = deepcopy(model.object_property_embeddings)

        model.add_axioms(self.axiom)
        model.train(epochs=0)
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
        model = SyntacticPlusW2VModel(self.dataset)

        with self.assertRaisesRegex(TypeError, "Parameter model must be a string pointing to the Word2Vec model file."):
            model.from_pretrained(1)

        with self.assertRaisesRegex(FileNotFoundError, "Pretrained model path does not exist"):
            model.from_pretrained("path")

        model2 = SyntacticPlusW2VModel(self.dataset)
        model2.set_w2v_model(min_count=1)
                    
    
    def test_train_after_pretrained(self):
        first_model = SyntacticPlusW2VModel(self.dataset, model_filepath="first_syntactic_w2v_model")
        first_model.set_w2v_model(min_count=1)
        first_model.generate_corpus(save=True, with_annotations=True)
        first_model.train()

        first_w2v_model = first_model.model_filepath
        first_model.w2v_model.save(first_w2v_model)
        
        self.assertTrue(os.path.exists(first_w2v_model))

        second_model = SyntacticPlusW2VModel(self.dataset)
        second_model.from_pretrained(first_w2v_model)

        self.assertNotEqual(second_model.model_filepath, first_w2v_model)

        second_model.generate_corpus(save=True, with_annotations=True)
        second_model.train(epochs=2)

