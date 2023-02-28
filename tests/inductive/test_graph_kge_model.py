from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.models import GraphPlusPyKEENModel
from mowl.projection import DL2VecProjector
from mowl.owlapi import OWLAPIAdapter
from copy import deepcopy
from pykeen.models import TransE
from mowl.utils.random import seed_everything


class TestKGEPyKEENModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
        self.projector = DL2VecProjector()
        
        adapter = OWLAPIAdapter()
        uncle = adapter.create_class("http://Uncle")
        has_sibling = adapter.create_object_property("http://hasSibling")
        father = adapter.create_class("http://Father")

        some_has_sibling = adapter.create_object_some_values_from(has_sibling, uncle)
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

        self.assertIn("http://Uncle", class_embeddings_after)

        for cls, emb in class_embeddings_before.items():
            with self.subTest(cls=cls):
                before = [round(x, 5) for x in emb]
                after = [round(x, 5) for x in class_embeddings_after[cls]]
                self.assertEqual(before, after)
                

        self.assertIn("http://hasSibling", property_embeddings_after)
        for prop, emb in property_embeddings_before.items():
            with self.subTest(prop=prop):
                self.assertEqual(emb.tolist(), property_embeddings_after[prop].tolist())


