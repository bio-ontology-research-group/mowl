from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.base_models import KGEModel, GraphModel
import pykeen
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
from mowl.projection import TaxonomyProjector
import mowl.error.messages as msg

class TestRandomWalkModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()


    def test_not_kge_method_set(self):
        """This should check that the method does not have a KGE method"""

        with self.assertRaisesRegex(AttributeError, msg.KGE_METHOD_NOT_SET):
            kge = KGEModel(self.dataset)
            kge.kge_method
        
    def test_instance_of_graph_model(self):
        model = KGEModel(self.dataset)
        self.assertIsInstance(model, GraphModel)
