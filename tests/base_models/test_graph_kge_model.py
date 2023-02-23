from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.base_models import KGEModel, GraphModel
import pykeen
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
from mowl.projection import TaxonomyProjector

class TestRandomWalkModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()


    def test_instance_of_graph_model(self):
        model = KGEModel(self.dataset)
        self.assertIsInstance(model, GraphModel)

    def test_get_triples_factory(self):
        model = KGEModel(self.dataset)
        model.set_projector(TaxonomyProjector())
        
        triples_factory = model.triples_factory
        self.assertIsInstance(triples_factory, TriplesFactory)
        
    def test_set_kge_method(self):
        """This should check the behaviour of the set_kge method"""

        model = KGEModel(self.dataset)
        model.set_projector(TaxonomyProjector())

        with self.assertRaisesRegex(TypeError,
                                    "Parameter 'kge_method' must be a pykeen.models.ERModel object"):
            model.set_kge_method(1)

            
        transe = TransE(triples_factory = model.triples_factory)
        model.set_kge_method(transe)
        self.assertIsInstance(model.kge_method, pykeen.models.ERModel)

