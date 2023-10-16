from tests.datasetFactory import FamilyDataset
from unittest import TestCase
import random
import os
import mowl
mowl.init_jvm("10g")
from mowl.base_models import Model


class TestModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()

    def test_model_parameter_types(self):
        """This checks if type of parameters is checked"""

        self.assertRaisesRegex(TypeError, "Parameter dataset must be a mOWL Dataset.", Model, 1)
        self.assertRaisesRegex(TypeError, "Optional parameter model_filepath must be of type str.",
                               Model, self.dataset, model_filepath=1)

    def test_train_method(self):
        """This checks if Model.train method works correctly"""

        model = Model(self.dataset)
        self.assertRaisesRegex(NotImplementedError, "Method train is not implemented.",
                               model.train)

    def test_eval_fn(self):
        """This checks if Model.eval_fn method works correctly"""

        model = Model(self.dataset)
        self.assertRaisesRegex(NotImplementedError, "Method eval_fn is not implemented.",
                               model.eval_fn)

    def test_get_class_index_dict_attribute(self):
        """This checks if Model.get_class_index_dict attribute works correctly"""

        model = Model(self.dataset)
        cls_id_dict = model.class_index_dict
        self.assertIsInstance(cls_id_dict, dict)
        rnd_idx = random.randrange(0, len(cls_id_dict))
        self.assertIsInstance(list(cls_id_dict.values())[rnd_idx], int)
        rnd_idx = random.randrange(0, len(cls_id_dict))
        self.assertIsInstance(list(cls_id_dict.keys())[rnd_idx], str)

    def test_get_object_property_index_dict_attribute(self):
        """This checks if Model.get_object_property_index_dict attribute works correctly"""

        model = Model(self.dataset)
        obj_prop_id_dict = model.object_property_index_dict
        self.assertIsInstance(obj_prop_id_dict, dict)
        rnd_idx = random.randrange(0, len(obj_prop_id_dict))
        self.assertIsInstance(list(obj_prop_id_dict.values())[rnd_idx], int)
        rnd_idx = random.randrange(0, len(obj_prop_id_dict))
        self.assertIsInstance(list(obj_prop_id_dict.keys())[rnd_idx], str)

    def test_get_model_filepath_attribute(self):
        """This checks if Model.model_filepath attribute works correctly"""

        model = Model(self.dataset, model_filepath="test")
        model_filepath = model.model_filepath
        self.assertIsInstance(model_filepath, str)
        self.assertEqual(model_filepath, "test")

    def test_temporary_model_filepath(self):
        """This checks temporary model filepath"""

        model = Model(self.dataset)
        model_filepath = model.model_filepath
        self.assertIsInstance(model_filepath, str)
        import tempfile
        tmppath = tempfile.gettempdir()
        self.assertTrue(model_filepath.startswith(tmppath))
