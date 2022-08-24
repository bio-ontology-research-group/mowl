"""
Test Cases for Dataset class and its subclasses
"""

from unittest import TestCase
from random import randrange
import os
import shutil
import requests
import mowl
mowl.init_jvm("10g")

from mowl.owlapi.model import OWLOntology, OWLClass, OWLObjectProperty
from mowl.datasets.builtin import PPIYeastSlimDataset, GDAHumanDataset
from mowl.datasets import PathDataset, RemoteDataset, TarFileDataset
from mowl.datasets.base import Entities, OWLClasses, OWLObjectProperties
from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import BOT, TOP

class TestPathDataset(TestCase):

    @classmethod
    def setUpClass(self):

        #Data for PathDataset
        self.ppi_dataset = PPIYeastSlimDataset()
        self.gda_dataset = GDAHumanDataset()
        self.training_ont_path = "ppi_yeast_slim/ontology.owl"
        self.validation_ont_path = "ppi_yeast_slim/valid.owl"
        self.testing_ont_path = "ppi_yeast_slim/test.owl"

        self.dataset_full = PathDataset(self.training_ont_path, self.validation_ont_path, self.testing_ont_path)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("ppi_yeast_slim")
        os.remove("ppi_yeast_slim.tar.gz")
        shutil.rmtree("gda_human")
        os.remove("gda_human.tar.gz")
        
    def test_file_not_found_exception(self):
        """It should return a FileNotFound error when file path not found"""
        self.assertRaises(FileNotFoundError, PathDataset, self.training_ont_path, validation_path = "")
        self.assertRaises(FileNotFoundError, PathDataset, "")
        self.assertRaises(FileNotFoundError, PathDataset, self.training_ont_path, testing_path = "")
        
    def test_constructor_argument_types(self):
        """It should handle incorrect argument types when constructing a path dataset"""
        self.assertRaises(TypeError, PathDataset, 1)
        self.assertRaises(TypeError, PathDataset, self.training_ont_path, validation_path = 1)
        self.assertRaises(TypeError, PathDataset, self.training_ont_path, testing_path = True)
        
    def test_type_of_ontology_attributes(self):
        """This should return the correct type of attributes"""

        dataset = self.dataset_full
        self.assertIsInstance(dataset.ontology, OWLOntology)
        self.assertIsInstance(dataset.testing, OWLOntology)
        self.assertIsInstance(dataset.validation, OWLOntology)

    def test_attribute_classes(self):
        """Test types of dataset.classes.as_owl attribute"""
        dataset = self.dataset_full
        classes = dataset.classes.as_owl
        self.assertIsInstance(classes, list)
        self.assertIsInstance(classes[0], OWLClass)

        

    def test_attribute_classes_as_str(self):
        """Test types and format of dataset.classes.as_str attribute"""
        dataset = self.dataset_full
        classes = dataset.classes.as_str
        #Type assertions
        self.assertIsInstance(classes, list)
        self.assertIsInstance(classes[0], str)

        #List must be sorted
        classes_copied = classes[:]
        classes_copied.sort()

        idx = randrange(0, len(classes_copied))
        self.assertEqual(classes[idx], classes_copied[idx])

        #Classes from OWLAPI should be the same as coming from dataset class.
        adapter = OWLAPIAdapter()
        classes_from_owl_api = list(dataset.ontology.getClassesInSignature()) + list(dataset.validation.getClassesInSignature()) + list(dataset.testing.getClassesInSignature()) + [adapter.create_class(TOP), adapter.create_class(BOT)]
        classes_from_owl_api = [str(x.toStringID()) for x in classes_from_owl_api]        
        classes_from_owl_api = list(set(classes_from_owl_api))
        classes_from_owl_api.sort()

        idx = randrange(0, len(classes_from_owl_api))
        self.assertEqual(classes[idx], classes_from_owl_api[idx])

    def test_attribute_object_properties(self):
        """Test types of dataset.object_properties.as_owl attribute"""
        dataset = self.dataset_full
        object_properties = dataset.object_properties.as_owl
        self.assertIsInstance(object_properties, list)
        self.assertIsInstance(object_properties[0], OWLObjectProperty)
                                

    def test_attribute_object_properties_as_str(self):
        """Test types and format of dataset.object_properties.as_str attribute"""
        dataset = PathDataset(self.training_ont_path, self.validation_ont_path, self.testing_ont_path)
        object_properties = dataset.object_properties.as_str
        #Type assertions
        self.assertIsInstance(object_properties, list)
        self.assertIsInstance(object_properties[0], str)

        #List must be sorted
        object_properties_copied = object_properties[:]
        object_properties_copied.sort()
        idx = randrange(0, len(object_properties_copied))
        self.assertEqual(object_properties[idx], object_properties_copied[idx])

        #Object_Properties from OWLAPI should be the same as coming from dataset class.
        adapter = OWLAPIAdapter()
        object_properties_from_owl_api = list(dataset.ontology.getObjectPropertiesInSignature()) + list(dataset.validation.getObjectPropertiesInSignature()) + list(dataset.testing.getObjectPropertiesInSignature())
        object_properties_from_owl_api = [str(x.toString())[1:-1] for x in object_properties_from_owl_api]        
        object_properties_from_owl_api = list(set(object_properties_from_owl_api))
        object_properties_from_owl_api.sort()

        idx = randrange(0, len(object_properties_from_owl_api))
        self.assertEqual(object_properties[idx], object_properties_from_owl_api[idx])

        
    
    def test_accessing_not_existing_attributes_is_None(self):
        """This checks if attributes can be accessed in any order"""
        dataset1 = PathDataset(self.training_ont_path, validation_path = None, testing_path = self.testing_ont_path)
        self.assertIsNone(dataset1.validation)

        dataset2 = PathDataset(self.training_ont_path, validation_path = self.validation_ont_path, testing_path = None)
        self.assertIsNone(dataset2.testing)

        dataset3 = PathDataset(self.training_ont_path, validation_path = None, testing_path = None)
        self.assertIsNone(dataset3.validation)
        self.assertIsNone(dataset3.testing)


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


    def test_label_property_existing(self):
        """It should check the correct behaviour of label property when it exists."""
        labels = self.ppi_dataset.labels
        self.assertIsInstance(labels, dict)
        idx = randrange(0, len(labels))
        self.assertIsInstance(list(labels.keys())[0], str)
        self.assertIsInstance(list(labels.values())[0], str)

        unique_labels = set(labels.keys())
        self.assertEqual(len(labels), len(unique_labels))
        
    def test_label_property_non_existing(self):
        """It should check return value to be {} when label property does not exist."""

        self.assertEqual({}, self.gda_dataset.labels)

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
        self.filepath = self.download(self, 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast_slim.tar.gz')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("/tmp/ppi_yeast_slim")
        os.remove("/tmp/ppi_yeast_slim.tar.gz")

    def test_extract_tar_file(self):
        """It should check correct extracting behaviour"""
        ds = TarFileDataset(self.filepath)

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


    def test_successful_download_in_default_path(self):
        """This checks if dataset is downloaded in the default path ./"""
        ds = RemoteDataset(self.good_url)
        self.assertTrue(os.path.exists("./ppi_yeast"))
        self.assertTrue(os.path.exists("./ppi_yeast/ontology.owl"))
        self.assertTrue(os.path.exists("./ppi_yeast/valid.owl"))
        self.assertTrue(os.path.exists("./ppi_yeast/test.owl"))

        shutil.rmtree("./ppi_yeast")
        os.remove("./ppi_yeast.tar.gz")

        
    def test_successful_download_in_custom_path(self):
        """This checks if dataset is downloaded a custom path"""
        ds = RemoteDataset(self.good_url, data_root = "/tmp/")
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
        ds = RemoteDataset(self.good_url, data_root = "/tmp/")
        file_timestamp1 = os.path.getmtime("/tmp/ppi_yeast.tar.gz")
        ds = RemoteDataset(self.good_url, data_root = "/tmp/")
        file_timestamp2 = os.path.getmtime("/tmp/ppi_yeast.tar.gz")

        self.assertEqual(file_timestamp1, file_timestamp2)

        shutil.rmtree("/tmp/ppi_yeast")
        os.remove("/tmp/ppi_yeast.tar.gz")

#############################################################

class TestEntities(TestCase):

    @classmethod
    def setUpClass(self):
        self.ds = PPIYeastSlimDataset()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./ppi_yeast_slim")
        os.remove("./ppi_yeast_slim.tar.gz")


    def test_method_check_owl_type_not_implemented(self):
        """This checks that NotImplementedError is raised for method check_owl_type"""

        empty = []
        self.assertRaises(NotImplementedError, Entities, empty)

    def test_type_for_classes_method(self):
        """This checks error handling when the OWLClasses class does not receive OWLClass objects"""

        props = self.ds.object_properties.as_owl
        self.assertRaises(TypeError, OWLClasses, props)

    def test_type_for_object_property_method(self):
        """This checks error handling when the OWLObjectProperties class does not receive OWLObjectProperty objects"""

        classes = self.ds.classes.as_owl
        self.assertRaises(TypeError, OWLObjectProperties, classes)
