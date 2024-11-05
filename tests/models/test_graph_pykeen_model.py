from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.models import GraphPlusPyKEENModel
from mowl.projection import TaxonomyProjector
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, ERModel
import mowl.error.messages as err
import torch as th

class TestPyKEENModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()

    def test_get_triples_factory(self):
        model = GraphPlusPyKEENModel(self.dataset)
        model.set_projector(TaxonomyProjector())
        
        triples_factory = model.triples_factory
        self.assertIsInstance(triples_factory, TriplesFactory)
        
    def test_set_kge_method(self):
        """This should check the behaviour of the set_kge method"""

        model = GraphPlusPyKEENModel(self.dataset)
        model.set_projector(TaxonomyProjector())

        with self.assertRaisesRegex(TypeError,
                                    "Parameter 'kge_method' must be a pykeen.models.ERModel object"):
            model.set_kge_method(1)

            
        transe = TransE
        model.set_kge_method(transe)
        self.assertIsInstance(model.kge_method, ERModel)


    def test_kge_method_not_set(self):
        """This checks if the exception is raised when the kge method is not set"""

        model = GraphPlusPyKEENModel(self.dataset)
        model.set_projector(TaxonomyProjector())
                        
        with self.assertRaisesRegex(AttributeError,
                                    err.PYKEEN_MODEL_NOT_SET):
            model.train(epochs=1)
        
    def test_get_embeddings(self):
        model = GraphPlusPyKEENModel(self.dataset)
        model.set_projector(TaxonomyProjector())
        
        with self.assertRaisesRegex(AttributeError,
                                    err.MODEL_NOT_TRAINED_OR_LOADED):
            class_embs = model.class_embeddings

        with self.assertRaisesRegex(AttributeError,
                                    err.MODEL_NOT_TRAINED_OR_LOADED):
            role_embs = model.object_property_embeddings
                

        with self.assertRaisesRegex(AttributeError,
                                    err.MODEL_NOT_TRAINED_OR_LOADED):
            ind_embs = model.individual_embeddings
            
        

        kge_method = TransE
        model.set_kge_method(kge_method)

        class_embs = model.class_embeddings
        self.assertIsInstance(class_embs, dict)
                
        role_embs = model.object_property_embeddings
        self.assertIsInstance(role_embs, dict)

        individual_embs = model.individual_embeddings
        self.assertIsInstance(individual_embs, dict)
        
