from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.models import GraphPlusPyKEENModel
from mowl.projection import DL2VecProjector
from mowl.owlapi import OWLAPIAdapter
from copy import deepcopy
from pykeen.models import TransE
from mowl.utils.random import seed_everything
import mowl.error.messages as msg
import torch as th
import os

class TestKGEPyKEENModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
        self.projector = DL2VecProjector()
        
        adapter = OWLAPIAdapter()
        aunt = adapter.create_class("http://Aunt")
        has_sibling = adapter.create_object_property("http://hasSibling")
        father = adapter.create_class("http://Father")

        some_has_sibling = adapter.create_object_some_values_from(has_sibling, aunt)
        self.axiom = adapter.create_subclass_of(father, some_has_sibling)

        
    def test_reindex_embeddings(self):
        seed_everything(42)
        model = GraphPlusPyKEENModel(self.dataset)
        model.set_projector(self.projector)
        model.set_kge_method(TransE, random_seed=42)
        class_embeddings_before = deepcopy(model.class_embeddings)
        property_embeddings_before = deepcopy(model.object_property_embeddings)

        
        model.add_axioms(self.axiom)
        
        class_embeddings_after = model.class_embeddings
        property_embeddings_after = model.object_property_embeddings

        self.assertIn("http://Aunt", class_embeddings_after)

        for cls, emb in class_embeddings_before.items():
            with self.subTest(cls=cls):
                before = [round(x, 5) for x in emb]
                after = [round(x, 5) for x in class_embeddings_after[cls]]
                self.assertEqual(before, after)
                

        self.assertIn("http://hasSibling", property_embeddings_after)
        for prop, emb in property_embeddings_before.items():
            with self.subTest(prop=prop):
                self.assertEqual(emb.tolist(), property_embeddings_after[prop].tolist())




                
    def test_from_pretrained(self):
        model = GraphPlusPyKEENModel(self.dataset)

        with self.assertRaisesRegex(TypeError, "Parameter model must be a string pointing to the PyKEEN model file."):
            model.from_pretrained(1)

        with self.assertRaisesRegex(FileNotFoundError, "Pretrained model path does not exist"):
            model.from_pretrained("path")

                                            

            
    def test_train_after_pretrained(self):
        first_model = GraphPlusPyKEENModel(self.dataset, model_filepath="first_kge_model")
        first_model.set_projector(self.projector)
        first_model.set_kge_method(TransE, random_seed=42)
        first_model.optimizer = th.optim.Adam
        first_model.lr = 0.001
        first_model.batch_size = 32
        first_model.train(epochs = 2)

        first_kge_model = first_model.model_filepath
        
        self.assertTrue(os.path.exists(first_kge_model))

        second_model = GraphPlusPyKEENModel(self.dataset)
        second_model.set_projector(self.projector)
        second_model.set_kge_method(TransE, random_seed=42)
        second_model.from_pretrained(first_kge_model)

        self.assertNotEqual(second_model.model_filepath, first_kge_model)
        second_model.set_projector(self.projector)
        second_model.optimizer = th.optim.Adam
        second_model.lr = 0.001
        second_model.batch_size = 32
        second_model.train(epochs=2)


