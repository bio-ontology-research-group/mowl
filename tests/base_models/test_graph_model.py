from unittest import TestCase

from tests.datasetFactory import FamilyDataset
from mowl.base_models import GraphModel
from mowl.projection import TaxonomyProjector
from mowl.base_models import Model

class TestGraphModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()


    def test_instance_of_model(self):
        """This should check if the graph model is an instance of Model"""

        model = GraphModel(self.dataset)
        self.assertIsInstance(model, Model)
        
    def test_set_projector_method(self):
        """This should check the behaviour of the set_projector method"""

        model = GraphModel(self.dataset)

        with self.assertRaisesRegex(TypeError,
                                    "Parameter 'projector' must be a mowl.projection.Projector object"):
            model.set_projector(1)

        model.set_projector(TaxonomyProjector())
        self.assertIsInstance(model.projector, TaxonomyProjector)
