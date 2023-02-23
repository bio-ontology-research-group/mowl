from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.base_models import RandomWalkModel, GraphModel
from mowl.walking import DeepWalk

class TestRandomWalkModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()


    def test_instance_of_graph_model(self):
        model = RandomWalkModel(self.dataset)
        self.assertIsInstance(model, GraphModel)

    def test_set_walking_method(self):
        """This should check the behaviour of the set_walker method"""

        model = RandomWalkModel(self.dataset)

        with self.assertRaisesRegex(TypeError,
                                    "Parameter 'walker' must be a mowl.walking.WalkingModel object"):
            model.set_walker(1)

        model.set_walker(DeepWalk(1,1))
        self.assertIsInstance(model.walker, DeepWalk)

