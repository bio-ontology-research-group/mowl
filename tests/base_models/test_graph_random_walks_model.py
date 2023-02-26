from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.base_models import RandomWalkModel, GraphModel, RandomWalkPlusW2VModel
from mowl.projection import TaxonomyProjector
from mowl.walking import DeepWalk
import mowl.error.messages as msg

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

    def test_rw_model_access_embeddings_error(self):
        """This should test that correct errors are raised when the RandomWalkModel is not trained"""
        model = RandomWalkModel(self.dataset)

        with self.assertRaises(NotImplementedError):
            model.class_embeddings

        with self.assertRaises(NotImplementedError):
            model.object_property_embeddings

        with self.assertRaises(NotImplementedError):
            model.individual_embeddings

    def test_rw_w2v_no_w2v_access_embeddings_error(self):
        """This should test that correct errors are raised when the RandomWalkPlusW2VModel does not have a w2v model"""
        model = RandomWalkPlusW2VModel(self.dataset)

        with self.assertRaisesRegex(AttributeError, msg.W2V_MODEL_NOT_SET):
            model.class_embeddings

        with self.assertRaisesRegex(AttributeError, msg.W2V_MODEL_NOT_SET):
            model.object_property_embeddings

        with self.assertRaisesRegex(AttributeError, msg.W2V_MODEL_NOT_SET):
            model.individual_embeddings

    def test_rw_w2v_not_trained_access_embeddings_error(self):
        """This should test that correct errors are raised when the RandomWalkPlusW2VModel is not trained"""
        model = RandomWalkPlusW2VModel(self.dataset)
        model.set_w2v_model()
        
        with self.assertRaisesRegex(AttributeError, msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND):
            model.class_embeddings

        with self.assertRaisesRegex(AttributeError, msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND):
            model.object_property_embeddings

        with self.assertRaisesRegex(AttributeError, msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND):
            model.individual_embeddings

            
    def test_train_method_error(self):
        """This should test that model cannot be trained without projector, walker set and w2v model set"""

        model = RandomWalkPlusW2VModel(self.dataset)
        with self.assertRaisesRegex(AttributeError, msg.GRAPH_MODEL_PROJECTOR_NOT_SET):
            model.train()

        model.set_projector(TaxonomyProjector())

        with self.assertRaisesRegex(AttributeError, msg.RANDOM_WALK_MODEL_WALKER_NOT_SET):
            model.train()

        model.set_walker(DeepWalk(1,1))

        with self.assertRaisesRegex(AttributeError, msg.W2V_MODEL_NOT_SET):
            model.train()
