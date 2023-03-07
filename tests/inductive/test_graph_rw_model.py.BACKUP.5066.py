from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.models import RandomWalkPlusW2VModel
from mowl.projection import DL2VecProjector
from mowl.walking import DeepWalk

from mowl.owlapi import OWLAPIAdapter
from copy import deepcopy
import mowl.error.messages as msg


class TestRandomWalkModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
        self.projector = DL2VecProjector()
        self.walker = DeepWalk(2,2)
        
        adapter = OWLAPIAdapter()
        uncle = adapter.create_class("http://Uncle")
        has_sibling = adapter.create_object_property("http://hasSibling")
        father = adapter.create_class("http://Father")

        some_has_sibling = adapter.create_object_some_values_from(has_sibling, uncle)
        self.axiom = adapter.create_subclass_of(father, some_has_sibling)
        

        
    def test_reindex_embeddings(self):
        model = RandomWalkPlusW2VModel(self.dataset)
        model.set_projector(self.projector)
        model.set_walker(self.walker)
        model.set_w2v_model(min_count=1)
        model.train()
        class_embeddings_before = deepcopy(model.class_embeddings)
        property_embeddings_before = deepcopy(model.object_property_embeddings)

        
        model.add_axioms(self.axiom)
        model.train(epochs=0)
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


<<<<<<< HEAD
    def test_pretrained_model(self):
        model = RandomWalkPlusW2VModel(self.dataset)
        model.set_projector(self.projector)
        model.set_walker(self.walker)
        class_embeddings_before = model
        


        model.add_axioms(self.axiom)
        
        model.train(epochs=0)
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
=======
    #def test_from_pretrained

    def test_from_pretrained(self):
        model = RandomWalkPlusW2VModel(self.dataset)

        with self.assertRaisesRegex(TypeError, "Pretrained model path must be a string"):
            model.from_pretrained(1)

        with self.assertRaisesRegex(FileNotFoundError, "Pretrained model path does not exist"):
            model.from_pretrained("path")

    
    def test_train_after_pretrained(self):
        model = RandomWalkPlusW2VModel(self.dataset)
        model.from_pretrained()

        with self.assertRaisesRegex(AttributeError, msg.GRAPH_MODEL_PROJECTOR_NOT_SET):
            model.train()

        model.set_projector(self.projector)

        with self.assertRaisesRegex(AttributeError, msg.RANDOM_WALK_MODEL_WALKER_NOT_SET):
            model.train()

        
        model.set_walker(self.walker)
        model.set_w2v_model(min_count=1)
        model.train()
        
>>>>>>> f4af22dbeafa773ecfe815fabcc39ba2efa3328b
