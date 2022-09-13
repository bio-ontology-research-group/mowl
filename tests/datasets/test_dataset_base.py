"""
Test Cases for Dataset class and its subclasses
"""

from mowl.owlapi.defaults import BOT, TOP
from mowl.owlapi import OWLAPIAdapter
from mowl.datasets.base import Entities, OWLClasses, OWLObjectProperties
from mowl.datasets import Dataset, PathDataset, RemoteDataset, TarFileDataset
from tests.datasetFactory import PPIYeastSlimDataset, GDAHumanELDataset
from mowl.owlapi.model import OWLOntology, OWLClass, OWLObjectProperty
from unittest import TestCase
from random import randrange, choice
import os
import shutil
import requests


class TestDataset(TestCase):

    @classmethod
    def setUpClass(self):
        adapter = OWLAPIAdapter()
        owl_manager = adapter.owl_manager
        self.training_ont = owl_manager.createOntology()
        self.validation_ont = owl_manager.createOntology()
        self.testing_ont = owl_manager.createOntology()

    def test_constructor_argument_types(self):
        """This checks if type of arguments is checked"""

        self.assertRaisesRegex(TypeError, "Parameter ontology must be an OWLOntology.", Dataset, 1)
        self.assertRaisesRegex(TypeError, "Optional parameter validation must be an OWLOntology.",
                               Dataset, self.training_ont, validation=1)
        self.assertRaisesRegex(TypeError, "Optional parameter testing must be an OWLOntology.",
                               Dataset, self.training_ont, testing=1)

    def test_accessing_not_existing_attributes_is_None(self):
        """This checks if attributes can be accessed in any order"""
        dataset1 = Dataset(self.training_ont, validation=None, testing=self.testing_ont)
        self.assertIsNone(dataset1.validation)

        dataset2 = Dataset(self.training_ont, validation=self.validation_ont, testing=None)
        self.assertIsNone(dataset2.testing)

        dataset3 = Dataset(self.training_ont, validation=None, testing=None)
        self.assertIsNone(dataset3.validation)
        self.assertIsNone(dataset3.testing)

    def test_label_property_non_existing(self):
        """It should check return value to be {} when label property does not exist."""
        dataset = Dataset(self.training_ont)
        self.assertEqual({}, dataset.labels)

###############################################################


class TestPathDataset(TestCase):

    @classmethod
    def setUpClass(self):

        # Data for PathDataset
        self.ppi_dataset = PPIYeastSlimDataset()
        self.gda_dataset = GDAHumanELDataset()
        self.training_ont_path = "ppi_yeast_slim/ontology.owl"
        self.validation_ont_path = "ppi_yeast_slim/valid.owl"
        self.testing_ont_path = "ppi_yeast_slim/test.owl"

        self.dataset_full = PathDataset(self.training_ont_path, self.validation_ont_path,
                                        self.testing_ont_path)

    def test_file_not_found_exception(self):
        """It should return a FileNotFound error when file path not found"""
        self.assertRaises(FileNotFoundError, PathDataset, self.training_ont_path,
                          validation_path="")
        self.assertRaises(FileNotFoundError, PathDataset, "")
        self.assertRaises(FileNotFoundError, PathDataset, self.training_ont_path,
                          testing_path="")

    def test_constructor_argument_types(self):
        """It should handle incorrect argument types when constructing a path dataset"""
        self.assertRaises(TypeError, PathDataset, 1)
        self.assertRaises(TypeError, PathDataset, self.training_ont_path, validation_path=1)
        self.assertRaises(TypeError, PathDataset, self.training_ont_path, testing_path=True)

    def test_type_of_ontology_attributes(self):
        """This should return the correct type of attributes"""

        dataset = self.dataset_full
        self.assertIsInstance(dataset.ontology, OWLOntology)
        self.assertIsInstance(dataset.testing, OWLOntology)
        self.assertIsInstance(dataset.validation, OWLOntology)

    def test_return_evaluation_classes_default(self):
        """It should check the default behaviour of evaluation classes property"""
        ds = self.dataset_full
        testing_classes_from_owlapi = ds.testing.getClassesInSignature()
        eval_classes = ds.evaluation_classes.as_str

        eval_classes_owl = [x.toString()[1:-1] for x in testing_classes_from_owlapi]
        eval_classes.sort()
        eval_classes_owl.sort()

        idx = randrange(0, len(eval_classes))
        self.assertEqual(eval_classes[idx], eval_classes_owl[idx])

    def test_attribute_classes(self):
        """Test types of dataset.classes.as_owl attribute"""
        dataset = self.dataset_full
        classes = dataset.classes.as_owl
        self.assertIsInstance(classes, list)
        self.assertIsInstance(classes[0], OWLClass)

    def test_attribute_object_properties(self):
        """Test types of dataset.object_properties.as_owl attribute"""
        dataset = self.dataset_full
        object_properties = dataset.object_properties.as_owl
        self.assertIsInstance(object_properties, list)
        self.assertIsInstance(object_properties[0], OWLObjectProperty)

    def test_label_property_existing(self):
        """It should check the correct behaviour of label property when it exists."""
        labels = self.ppi_dataset.labels
        self.assertIsInstance(labels, dict)
        self.assertIsInstance(list(labels.keys())[0], str)
        self.assertIsInstance(list(labels.values())[0], str)

        unique_labels = set(labels.keys())
        self.assertEqual(len(labels), len(unique_labels))

    def test_attribute_classes_as_str(self):
        """Test types and format of dataset.classes.as_str attribute"""
        dataset = self.dataset_full
        classes = dataset.classes.as_str
        # Type assertions
        self.assertIsInstance(classes, list)
        self.assertIsInstance(classes[0], str)

        # List must be sorted
        classes_copied = classes[:]
        classes_copied.sort()

        idx = randrange(0, len(classes_copied))
        self.assertEqual(classes[idx], classes_copied[idx])

        # Classes from OWLAPI should be the same as coming from dataset class.
        adapter = OWLAPIAdapter()
        classes_from_owl_api = list(dataset.ontology.getClassesInSignature())
        classes_from_owl_api += list(dataset.validation.getClassesInSignature())
        classes_from_owl_api += list(dataset.testing.getClassesInSignature())
        classes_from_owl_api += [adapter.create_class(TOP), adapter.create_class(BOT)]

        classes_from_owl_api = [str(x.toStringID()) for x in classes_from_owl_api]
        classes_from_owl_api = list(set(classes_from_owl_api))
        classes_from_owl_api.sort()

        idx = randrange(0, len(classes_from_owl_api))
        self.assertEqual(classes[idx], classes_from_owl_api[idx])

    def test_attribute_object_properties_as_str(self):
        """Test types and format of dataset.object_properties.as_str attribute"""
        dataset = PathDataset(self.training_ont_path, self.validation_ont_path,
                              self.testing_ont_path)
        object_properties = dataset.object_properties.as_str
        # Type assertions
        self.assertIsInstance(object_properties, list)
        self.assertIsInstance(object_properties[0], str)

        # List must be sorted
        object_properties_copied = object_properties[:]
        object_properties_copied.sort()
        idx = randrange(0, len(object_properties_copied))
        self.assertEqual(object_properties[idx], object_properties_copied[idx])

        # Object_Properties from OWLAPI should be the same as coming from dataset class.
        object_properties_from_owl_api = list(dataset.ontology.getObjectPropertiesInSignature())
        object_properties_from_owl_api += list(dataset.validation.getObjectPropertiesInSignature())
        object_properties_from_owl_api += list(dataset.testing.getObjectPropertiesInSignature())
        object_properties_from_owl_api = [str(x.toString())[1:-1] for x in
                                          object_properties_from_owl_api]
        object_properties_from_owl_api = list(set(object_properties_from_owl_api))
        object_properties_from_owl_api.sort()

        idx = randrange(0, len(object_properties_from_owl_api))
        self.assertEqual(object_properties[idx], object_properties_from_owl_api[idx])


#############################################################

class TestTarFileDataset(TestCase):

    def download(self, url):
        filename = url.split('/')[-1]
        filepath = os.path.join("/tmp/", filename)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filepath

    @classmethod
    def setUpClass(self):
        self.filepath = self.download(
            self,
            'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast_slim.tar.gz')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("/tmp/ppi_yeast_slim")
        os.remove("/tmp/ppi_yeast_slim.tar.gz")

    def test_extract_tar_file(self):
        """It should check correct extracting behaviour"""
        _ = TarFileDataset(self.filepath)

        self.assertTrue(os.path.exists("/tmp/ppi_yeast_slim.tar.gz"))
        self.assertTrue(os.path.exists("/tmp/ppi_yeast_slim/ontology.owl"))
        self.assertTrue(os.path.exists("/tmp/ppi_yeast_slim/valid.owl"))
        self.assertTrue(os.path.exists("/tmp/ppi_yeast_slim/test.owl"))

#############################################################


class TestRemoteDataset(TestCase):

    @classmethod
    def setUpClass(self):

        self.good_url = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast.tar.gz'
        self.bad_url = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_mouse_el.tar.gzq'
        self.only_training_set_url = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/family.tar.gz'

    def test_successful_download_in_default_path(self):
        """This checks if dataset is downloaded in the default path ./"""
        _ = RemoteDataset(self.good_url)
        self.assertTrue(os.path.exists("./ppi_yeast"))
        self.assertTrue(os.path.exists("./ppi_yeast/ontology.owl"))
        self.assertTrue(os.path.exists("./ppi_yeast/valid.owl"))
        self.assertTrue(os.path.exists("./ppi_yeast/test.owl"))

    def test_successful_download_in_custom_path(self):
        """This checks if dataset is downloaded a custom path"""
        _ = RemoteDataset(self.good_url, data_root="/tmp/")
        self.assertTrue(os.path.exists("/tmp/ppi_yeast"))
        self.assertTrue(os.path.exists("/tmp/ppi_yeast/ontology.owl"))
        self.assertTrue(os.path.exists("/tmp/ppi_yeast/valid.owl"))
        self.assertTrue(os.path.exists("/tmp/ppi_yeast/test.owl"))

        shutil.rmtree("/tmp/ppi_yeast")
        os.remove("/tmp/ppi_yeast.tar.gz")

    def test_incorrect_url(self):
        """This checks if error is raised for incorrect URL"""
        self.assertRaises(requests.exceptions.HTTPError, RemoteDataset, self.bad_url)

    def test_dataset_not_downloaded_if_already_exists(self):
        """This should check that dataset is not downloaded if already exists"""
        _ = RemoteDataset(self.good_url, data_root="/tmp/")
        file_timestamp1 = os.path.getmtime("/tmp/ppi_yeast.tar.gz")
        _ = RemoteDataset(self.good_url, data_root="/tmp/")
        file_timestamp2 = os.path.getmtime("/tmp/ppi_yeast.tar.gz")

        self.assertEqual(file_timestamp1, file_timestamp2)

        shutil.rmtree("/tmp/ppi_yeast")
        os.remove("/tmp/ppi_yeast.tar.gz")

    def test_dataset_with_only_training_set(self):
        """This should check that dataset is downloaded correctly if it has only training set"""
        _ = RemoteDataset(self.only_training_set_url, data_root="/tmp/")
        self.assertTrue(os.path.exists("/tmp/family"))
        self.assertTrue(os.path.exists("/tmp/family/ontology.owl"))
        self.assertFalse(os.path.exists("/tmp/family/valid.owl"))
        self.assertFalse(os.path.exists("/tmp/family/test.owl"))

        shutil.rmtree("/tmp/family")
        os.remove("/tmp/family.tar.gz")
#############################################################


class TestEntities(TestCase):

    @classmethod
    def setUpClass(self):
        self.ds = PPIYeastSlimDataset()

    def test_method_check_owl_type_not_implemented(self):
        """This checks that NotImplementedError is raised for method check_owl_type"""

        empty = []
        self.assertRaises(NotImplementedError, Entities, empty)

    def test_type_for_classes_method(self):
        """This checks error handling when the OWLClasses class does not receive OWLClass \
            objects"""

        props = self.ds.object_properties.as_owl
        self.assertRaises(TypeError, OWLClasses, props)

    def test_type_for_object_property_method(self):
        """This checks error handling when the OWLObjectProperties class does not receive \
            OWLObjectProperty objects"""

        classes = self.ds.classes.as_owl
        self.assertRaises(TypeError, OWLObjectProperties, classes)

    def test_format_of_class_as_str(self):
        """This checks if the format of the class string is correct"""

        classes = self.ds.classes.as_str
        owl_class_str = choice(classes)
        self.assertFalse(owl_class_str.startswith("<"))
        self.assertFalse(owl_class_str.endswith(">"))
        self.assertTrue(owl_class_str.startswith("http://"))

    def test_format_of_object_property_as_str(self):
        """This checks if the format of the object property string is correct"""

        props = self.ds.object_properties.as_str
        owl_prop_str = choice(props)
        self.assertFalse(owl_prop_str.startswith("<"))
        self.assertFalse(owl_prop_str.endswith(">"))
        self.assertTrue(owl_prop_str.startswith("http://"))
