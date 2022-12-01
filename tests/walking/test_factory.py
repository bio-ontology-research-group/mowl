import mowl.error as err
from mowl.walking import DeepWalk, Node2Vec, walker_factory
from unittest import TestCase


class TestFactoryMethod(TestCase):

    def test_factory_return_types(self):
        """This method test if the return walkers from factory method are correct"""

        # Test DeepWalk
        self.assertIsInstance(walker_factory("deepwalk", 1, 1), DeepWalk)

        # Test Node2Vec
        self.assertIsInstance(walker_factory("node2vec", 1, 1), Node2Vec)

        # Test if exception is raised when walker name is not valid
        self.assertRaisesRegex(ValueError, err.INVALID_WALKER_NAME, walker_factory,
                               "invalid_walker_name", 1, 1)
