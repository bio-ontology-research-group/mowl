"""
Test Cases for Dataset class and its subclasses
"""

from mowl.owlapi.defaults import BOT, TOP
from mowl.owlapi import OWLAPIAdapter
from mowl.datasets.base import Entities, OWLClasses, OWLObjectProperties, OWLIndividuals, \
    DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT, DOWNLOAD_RETRIES
from mowl.datasets import Dataset, PathDataset, RemoteDataset, TarFileDataset, default_data_root
from tests.datasetFactory import PPIYeastSlimDataset, GDAHumanELDataset, FamilyDataset, data_path
from mowl.owlapi.model import OWLOntology, OWLClass, OWLObjectProperty
from unittest import TestCase, mock
from random import randrange, choice
import os
import shutil
import requests
import tempfile

from org.semanticweb.owlapi.model import IRI


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
        self.training_ont_path = data_path("ppi_yeast_slim/ontology.owl")
        self.validation_ont_path = data_path("ppi_yeast_slim/valid.owl")
        self.testing_ont_path = data_path("ppi_yeast_slim/test.owl")

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

        with self.assertRaisesRegex(NotImplementedError, "This method must be implemented in a subclass."):
            ds.evaluation_classes

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
        filepath = os.path.join(tempfile.gettempdir(), filename)

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
            'https://bio2vec.net/data/mowl/ppi_yeast_slim.tar.gz')

    @classmethod
    def tearDownClass(self):
        tmp_dir = tempfile.gettempdir()
        shutil.rmtree(os.path.join(tmp_dir, "ppi_yeast_slim"))
        os.remove(os.path.join(tmp_dir, "ppi_yeast_slim.tar.gz"))

    def test_extract_tar_file(self):
        """It should check correct extracting behaviour"""
        _ = TarFileDataset(self.filepath)
        tmp_dir = tempfile.gettempdir()
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast_slim.tar.gz")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast_slim/ontology.owl")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast_slim/valid.owl")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast_slim/test.owl")))

#############################################################


class TestRemoteDataset(TestCase):

    @classmethod
    def setUpClass(self):

        self.good_url = 'https://bio2vec.net/data/mowl/ppi_yeast.tar.gz'
        self.bad_url = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_mouse_el.tar.gzq'
        self.only_training_set_url = 'https://bio2vec.net/data/mowl/family.tar.gz'

        tmp_dir = tempfile.gettempdir()
        self.tmp_dir = os.path.join(tmp_dir, "mowl")
        os.makedirs(self.tmp_dir, exist_ok=True)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tmp_dir)

    def setUp(self):
        for file_ in os.listdir(self.tmp_dir):
            if os.path.isfile(os.path.join(self.tmp_dir, file_)):
                os.remove(os.path.join(self.tmp_dir, file_))
            elif os.path.isdir(os.path.join(self.tmp_dir, file_)):
                shutil.rmtree(os.path.join(self.tmp_dir, file_))

    def test_successful_download_in_default_path(self):
        """This checks that the dataset lands in the default cache, not the working directory"""
        tmp_dir = self.tmp_dir
        with mock.patch.dict(os.environ, {'MOWL_DATA_ROOT': tmp_dir}):
            _ = RemoteDataset(self.good_url)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast/ontology.owl")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast/valid.owl")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast/test.owl")))
        # That the default is not the working directory is asserted in
        # TestDefaultDataRoot; a checkout may already hold a stale ./ppi_yeast
        # from before the cache existed, so it cannot be checked from here.

    def test_successful_download_in_custom_path(self):
        """This checks if dataset is downloaded a custom path"""
        tmp_dir = self.tmp_dir
        _ = RemoteDataset(self.good_url, data_root=tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast/ontology.owl")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast/valid.owl")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ppi_yeast/test.owl")))

    def test_incorrect_url(self):
        """This checks if error is raised for incorrect URL"""
        self.assertRaises(requests.exceptions.HTTPError, RemoteDataset, self.bad_url,
                          self.tmp_dir)

    def test_no_partial_file_left_behind_on_failure(self):
        """A download that fails must not leave a truncated file behind"""
        tmp_dir = self.tmp_dir
        self.assertRaises(requests.exceptions.HTTPError, RemoteDataset, self.bad_url, tmp_dir)
        leftovers = [f for f in os.listdir(tmp_dir) if f.startswith("gda_mouse_el.tar.gz")]
        self.assertEqual(leftovers, [])

    def test_interrupted_download_is_not_reused(self):
        """A transfer that dies mid-stream leaves no file the next run would trust"""
        tmp_dir = self.tmp_dir
        filename = self.good_url.split('/')[-1]

        def die_after_one_chunk(*args, **kwargs):
            yield b'partial payload'
            raise requests.exceptions.ReadTimeout("connection went silent")

        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.side_effect = die_after_one_chunk
        session = mock.MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = response

        with mock.patch.object(RemoteDataset, '_session', return_value=session):
            self.assertRaises(requests.exceptions.ReadTimeout,
                              RemoteDataset, self.good_url, tmp_dir)

        leftovers = [f for f in os.listdir(tmp_dir) if f.startswith(filename)]
        self.assertEqual(leftovers, [])

    def test_download_passes_a_timeout(self):
        """``requests`` has no default timeout, so the call must set one explicitly"""
        # Drive _download directly: the point is the arguments of the request,
        # not the extraction that follows it.
        dataset = RemoteDataset.__new__(RemoteDataset)
        dataset.url = self.good_url
        dataset.data_root = self.tmp_dir

        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b'payload']
        session = mock.MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = response

        with mock.patch.object(RemoteDataset, '_session', return_value=session):
            dataset._download()

        _, kwargs = session.get.call_args
        self.assertEqual(kwargs['timeout'],
                         (DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT))

    def test_download_is_retried(self):
        """The session must mount an adapter that retries stalled transfers"""
        adapter = RemoteDataset._session().get_adapter('https://bio2vec.net/')
        self.assertEqual(adapter.max_retries.total, DOWNLOAD_RETRIES)
        self.assertGreater(adapter.max_retries.backoff_factor, 0)

    def test_dataset_not_downloaded_if_already_exists(self):
        """This should check that dataset is not downloaded if already exists"""
        tmp_dir = self.tmp_dir
        _ = RemoteDataset(self.good_url, data_root=tmp_dir)
        file_timestamp1 = os.path.getmtime(os.path.join(tmp_dir, "ppi_yeast.tar.gz"))
        _ = RemoteDataset(self.good_url, data_root=tmp_dir)
        file_timestamp2 = os.path.getmtime(os.path.join(tmp_dir, "ppi_yeast.tar.gz"))

        self.assertEqual(file_timestamp1, file_timestamp2)

    def test_dataset_with_only_training_set(self):
        """This should check that dataset is downloaded correctly if it has only training set"""
        tmp_dir = self.tmp_dir
        _ = RemoteDataset(self.only_training_set_url, data_root=tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "family")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "family/ontology.owl")))
        self.assertFalse(os.path.exists(os.path.join(tmp_dir, "family/valid.owl")))
        self.assertFalse(os.path.exists(os.path.join(tmp_dir, "family/test.owl")))

#############################################################


class TestDefaultDataRoot(TestCase):
    """Resolution of the directory remote datasets are cached in. No network."""

    def _resolve(self, **env):
        """Resolve the default root with only ``env`` set among the vars we read."""
        patched = {'MOWL_DATA_ROOT': '', 'XDG_CACHE_HOME': ''}
        patched.update(env)
        with mock.patch.dict(os.environ, patched):
            for name, value in patched.items():
                if not value:
                    os.environ.pop(name, None)
            return default_data_root()

    def test_is_not_the_working_directory(self):
        """The regression this guards: datasets used to land wherever python was run"""
        root = self._resolve()
        self.assertNotEqual(os.path.abspath(root), os.path.abspath('.'))
        self.assertTrue(os.path.isabs(root))

    def test_mowl_data_root_wins(self):
        root = self._resolve(MOWL_DATA_ROOT='/custom/place', XDG_CACHE_HOME='/xdg')
        self.assertEqual(root, '/custom/place')

    def test_mowl_data_root_expands_user(self):
        root = self._resolve(MOWL_DATA_ROOT='~/somewhere')
        self.assertEqual(root, os.path.join(os.path.expanduser('~'), 'somewhere'))

    def test_xdg_cache_home_is_honoured(self):
        root = self._resolve(XDG_CACHE_HOME='/xdg')
        self.assertEqual(root, os.path.join('/xdg', 'mowl', 'datasets'))

    def test_falls_back_to_home_cache(self):
        root = self._resolve()
        self.assertEqual(root, os.path.join(os.path.expanduser('~'), '.cache', 'mowl',
                                            'datasets'))

#############################################################


class TestEntities(TestCase):

    @classmethod
    def setUpClass(self):
        self.ds = PPIYeastSlimDataset()
        self.family_ds = FamilyDataset()

        adapter = OWLAPIAdapter()
        manager = adapter.owl_manager
        data_factory = adapter.data_factory

        individual = data_factory.getOWLNamedIndividual(IRI.create("http://Jhon"))
        person = data_factory.getOWLClass(IRI.create("http://Person"))
        assertion = data_factory.getOWLClassAssertionAxiom(person, individual)
        manager.addAxiom(self.family_ds.ontology, assertion)

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

    def test_type_for_individuals_method(self):
        """This check error handling when the OWLIndividuals class does not receive \
OWLIndividual objects"""

        classes = self.ds.classes.as_owl
        self.assertRaises(TypeError, OWLIndividuals, classes)

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

    def test_format_of_individual_as_str(self):
        """This checks if the format of the individual string is correct"""

        individuals = self.family_ds.individuals.as_str
        owl_individual_str = choice(individuals)
        self.assertFalse(owl_individual_str.startswith("<"))
        self.assertFalse(owl_individual_str.endswith(">"))
        self.assertTrue(owl_individual_str.startswith("http://"))
