"""
This module contains classes intended to deal with mOWL datasets.
"""

import tarfile
import pathlib
import os

from jpype import java
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# OWLAPI imports
from org.semanticweb.owlapi.model import OWLOntology, OWLClass, OWLObjectProperty, OWLIndividual
from org.semanticweb.owlapi.apibinding import OWLManager

from mowl.projection import TaxonomyWithRelationsProjector
from mowl.owlapi.adapter import OWLAPIAdapter
from mowl.owlapi.defaults import TOP, BOT
from deprecated.sphinx import versionadded, versionchanged


from java.util import HashSet

#: Seconds to wait for the connection to the dataset host to be established.
DOWNLOAD_CONNECT_TIMEOUT = 10
#: Seconds to wait between two chunks of the response body. ``requests`` has no
#: default timeout, so without this a socket that goes silent mid-transfer
#: blocks forever instead of failing.
DOWNLOAD_READ_TIMEOUT = 60
#: How many times a stalled or server-side-failed download is retried.
DOWNLOAD_RETRIES = 3


def default_data_root():
    """Directory where remote datasets are cached.

    ``MOWL_DATA_ROOT`` overrides everything. Otherwise this follows the XDG
    base directory specification: ``$XDG_CACHE_HOME/mowl/datasets``, falling
    back to ``~/.cache/mowl/datasets``.

    :rtype: str
    """
    env_root = os.environ.get('MOWL_DATA_ROOT')
    if env_root:
        return os.path.expanduser(env_root)

    cache_home = os.environ.get('XDG_CACHE_HOME')
    if not cache_home:
        cache_home = os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(cache_home, 'mowl', 'datasets')


class Dataset():
    """This class represents an mOWL dataset.

    :param ontology: The ontology containing the training data of the dataset.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param validation: The ontology containing the validation data of the dataset, defaults to \
    ``None``.
    :type validation: :class:`org.semanticweb.owlapi.model.OWLOntology`, optional
    :param testing: The ontology containing the testing data of the dataset, defaults to ``None``.
    :type testing: :class:`org.semanticweb.owlapi.model.OWLOntology`, optional
    """

    def __init__(self, ontology, validation=None, testing=None):

        if not isinstance(ontology, OWLOntology):
            raise TypeError("Parameter ontology must be an OWLOntology.")

        if not isinstance(validation, OWLOntology) and validation is not None:
            raise TypeError("Optional parameter validation must be an OWLOntology.")

        if not isinstance(testing, OWLOntology) and testing is not None:
            raise TypeError("Optional parameter testing must be an OWLOntology.")

        self._ontology = ontology
        self._validation = validation
        self._testing = testing

        self._classes = None
        self._individuals = None
        self._object_properties = None
        self._individuals = None
        self._evaluation_classes = None

    @property
    def ontology(self):
        """Training dataset

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        return self._ontology

    @property
    def validation(self):
        """Validation dataset

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        return self._validation

    @property
    def testing(self):
        """Testing ontology

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        return self._testing

    @property
    def classes(self):
        """List of classes in the dataset. The classes are collected from training, validation and
        testing ontologies using the OWLAPI method ``ontology.getClassesInSignature()``.

        :rtype: OWLClasses
        """
        if self._classes is None:
            adapter = OWLAPIAdapter()
            top = adapter.create_class(TOP)
            bot = adapter.create_class(BOT)
            classes = set([top, bot])
            classes |= set(self._ontology.getClassesInSignature())

            if self._validation:
                classes |= set(self._validation.getClassesInSignature())
            if self._testing:
                classes |= set(self._testing.getClassesInSignature())

            classes = list(classes)
            self._classes = OWLClasses(classes)
        return self._classes

    @property
    def class_to_id(self):
        """
        Dictionary mapping :class:`OWLClasses <mowl.datasets.OWLClasses>` to integer ids.
        """
        return self.classes.as_index_dict

    @property
    def individuals(self):
        """List of individuals in the dataset. The individuals are collected from training, \
validation and testing ontologies using the OWLAPI method ``ontology.getIndividualsSignature()``.

        :rtype: OWLIndividuals
        """
        if self._individuals is None:
            individuals = set(self._ontology.getIndividualsInSignature())
            if self._validation:
                individuals |= set(self._validation.getIndividualsInSignature())
            if self._testing:
                individuals |= set(self._testing.getIndividualsInSignature())
            individuals = list(individuals)
            self._individuals = OWLIndividuals(individuals)
        return self._individuals

    @property
    def individual_to_id(self):
        """
        Dictionary mapping :class:`OWLIndividuals <mowl.datasets.OWLIndividuals>` to integer ids.
        """

        return self.individuals.as_index_dict

    @property
    def object_properties(self):
        """List of object properties (relations) in the dataset. The object
        properties are collected from training, validation and testing
        ontologies using the OWLAPI
        method ``ontology.getObjectPropertiesInSignature()``.

        :rtype: OWLObjectProperties
        """

        if self._object_properties is None:
            obj_properties = set()
            obj_properties |= set(self._ontology.getObjectPropertiesInSignature())

            if self._validation:
                obj_properties |= set(self._validation.getObjectPropertiesInSignature())
            if self._testing:
                obj_properties |= set(self._testing.getObjectPropertiesInSignature())

            obj_properties = list(obj_properties)
            self._object_properties = OWLObjectProperties(obj_properties)
        return self._object_properties

    @property
    def object_property_to_id(self):
        """
        Dictionary mapping :class:`OWLObjectProperties <mowl.datasets.OWLObjectProperties>` to integer ids.
        """
        return self.object_properties.as_index_dict


    @versionchanged(version='0.4.0', reason='Delegate implementation to subclasses.')
    @property
    def evaluation_classes(self):
        """Pair of lists of classes used for evaluation.  The return type is a tuple \
        of :class:`OWLClasses` objects.

        :rtype: tuple
        """

        raise NotImplementedError("This method must be implemented in a subclass.")
                                        

    @property
    def labels(self):
        """This method returns labels of entities as a dictionary. To be
        called, the training ontology must contain axioms of the form
        :math:`class_1 \sqsubseteq \exists http://has\_label . class_2`.

        :rtype: dict
        """
        projector = TaxonomyWithRelationsProjector(relations=["http://has_label"])
        edges = projector.project(self._ontology)
        labels = {str(e.src): str(e.dst) for e in edges}
        return labels


    @versionadded(version="0.2.0")
    def add_axioms(self, *axioms):
        manager = OWLAPIAdapter().owl_manager
        axiom_set = HashSet(axioms)
        manager.addAxioms(self.ontology, axiom_set)

        self._classes = None
        self._individuals = None
        self._object_properties = None
        self._individuals = None
        self._evaluation_classes = None

        

    
class PathDataset(Dataset):
    """Loads the dataset from ontology documents.

    :param ontology_path: Training dataset
    :type ontology_path: str
    :param validation_path: Validation dataset. Defaults to ``None``.
    :type validation_path: str, optional
    :param testing_path: Testing dataset. Defaults to ``None``.
    :type testing_path: str, optional
    """

    ontology_path: str
    validation_path: str
    testing_path: str
    _ontology: OWLOntology
    _validation: OWLOntology
    _testing: OWLOntology

    def __init__(self,
                 ontology_path: str,
                 validation_path: str = None,
                 testing_path: str = None):

        # Checks on training file path
        if not isinstance(ontology_path, str):
            raise TypeError("Training ontology path must be a string.")

        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Training ontology file not found {ontology_path}")

        # Checks on validation file path
        if validation_path is not None:
            if not isinstance(validation_path, str):
                raise TypeError("Training validation path must be a string.")

            if not os.path.exists(validation_path):
                raise FileNotFoundError(f"Validation ontology file not found {validation_path}")

        # Checks on testing file path
        if testing_path is not None:
            if not isinstance(testing_path, str):
                raise TypeError("Training testing path must be a string.")

            if not os.path.exists(testing_path):
                raise FileNotFoundError(f"Testing ontology file not found {testing_path}")

        self.ontology_path = ontology_path
        self.validation_path = validation_path
        self.testing_path = testing_path

        ontology, validation, testing = self._load()
        super().__init__(ontology, validation=validation, testing=testing)

        self._loaded = False
        self._classes = None
        self._object_properties = None
        self._evaluation_classes = None

    def _load(self):

        ont_manager = OWLManager.createOWLOntologyManager()
        ontology = ont_manager.loadOntologyFromOntologyDocument(
            java.io.File(self.ontology_path))

        validation = None
        if self.validation_path is not None:
            validation = ont_manager.loadOntologyFromOntologyDocument(
                java.io.File(self.validation_path))

        testing = None
        if self.testing_path is not None:
            testing = ont_manager.loadOntologyFromOntologyDocument(
                java.io.File(self.testing_path))

        return ontology, validation, testing


class TarFileDataset(PathDataset):
    """Loads the dataset from a `tar` file.

    :param tarfile_path: Location of the `tar` file
    :type tarfile_path: str

    :param \**kwargs:
        See below
    :Keyword Arguments:
        * **dataset_name** (str): Name of the dataset
    """
    tarfile_path: str
    dataset_name: str
    data_root: str

    def __init__(self, tarfile_path: str, *args, **kwargs):
        self.tarfile_path = tarfile_path
        self.dataset_name = kwargs.pop('dataset_name', None)
        if self.dataset_name is None:
            basename = os.path.basename(self.tarfile_path)
            self.dataset_name = basename.split(os.extsep, 1)[0]
        self.data_root = pathlib.Path(self.tarfile_path).parent

        dataset_root = os.path.join(self.data_root, self.dataset_name)
        self.root = dataset_root

        ontology_path = os.path.join(dataset_root, 'ontology.owl')
        validation_path = os.path.join(dataset_root, 'valid.owl')
        testing_path = os.path.join(dataset_root, 'test.owl')

        ontology_exists = os.path.exists(ontology_path)
        validation_exists = os.path.exists(validation_path)
        testing_exists = os.path.exists(testing_path)

        # Check if the dataset is already extracted
        if not (ontology_exists and validation_exists and testing_exists):
            self._extract()

        # Check if validation and testing ontologies exist
        if not os.path.exists(validation_path):
            validation_path = None
        if not os.path.exists(testing_path):
            testing_path = None

        super().__init__(
            ontology_path,
            validation_path,
            testing_path)

    def _extract(self):
        with tarfile.open(self.tarfile_path) as tarf:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tarf, path=self.data_root)


@versionchanged(
    reason="Datasets are cached under :func:`default_data_root` instead of the current "
           "working directory, and downloads have a timeout, retries and are atomic.",
    version="2.2.0")
class RemoteDataset(TarFileDataset):
    """Loads the dataset from a remote URL.

    The dataset is downloaded once and reused on later instantiations.

    :param url: URL location of the dataset
    :type url: str
    :param data_root: Directory the dataset is downloaded into and extracted \
        in. Defaults to :func:`default_data_root`, a user-level cache shared \
        by every working directory.
    :type data_root: str, optional
    """

    url: str
    data_root: str

    def __init__(self, url: str, data_root=None):
        self.url = url
        self.data_root = default_data_root() if data_root is None else data_root
        os.makedirs(self.data_root, exist_ok=True)
        tarfile_path = self._download()
        super().__init__(tarfile_path)

    @staticmethod
    def _session():
        """A :class:`requests.Session` that retries stalled downloads with backoff."""
        retry = Retry(total=DOWNLOAD_RETRIES, backoff_factor=1,
                      status_forcelist=(429, 500, 502, 503, 504))
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        return session

    def _download(self):
        filename = self.url.split('/')[-1]
        filepath = os.path.join(self.data_root, filename)
        if os.path.exists(filepath):
            return filepath

        # Write to a temporary name and rename only once the transfer finished,
        # so an interrupted download cannot leave a truncated file that the
        # check above would later mistake for a complete one. The name carries
        # the pid so that two processes sharing a cache cannot write to the
        # same partial file; os.replace makes the last one to finish win.
        partial_path = '{}.{}.part'.format(filepath, os.getpid())
        timeout = (DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT)
        try:
            with self._session() as session:
                with session.get(self.url, stream=True, timeout=timeout) as req:
                    req.raise_for_status()
                    with open(partial_path, 'wb') as writer:
                        for chunk in req.iter_content(chunk_size=8192):
                            writer.write(chunk)
        except BaseException:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            raise

        os.replace(partial_path, filepath)
        return filepath


class Entities():
    """Abstract class containing OWLEntities indexed by they IRIs"""

    def __init__(self, collection):
        self._collection = self.check_owl_type(collection)
        self._collection = sorted(self._collection, key=lambda x: x.toStringID())
        self._name_owlobject = self.to_dict()
        self._index_dict = self.to_index_dict()

    def __getitem__(self, idx):
        return self._collection[idx]

    def __len__(self):
        return len(self._collection)

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind < len(self._collection):
            item = self._collection[self.ind]
            self.ind += 1
            return item
        raise StopIteration

    def check_owl_type(self, collection):
        """This method checks whether the elements in the provided collection
        are of the correct type.
        """
        raise NotImplementedError

    def to_str(self, owl_class):
        raise NotImplementedError

    def to_dict(self):
        """Generates a dictionaty indexed by OWL entities IRIs and the values
        are the corresponding OWL entities.
        """
        dict_ = {self.to_str(ent): ent for ent in self._collection}
        return dict_

    def to_index_dict(self):
        """Generates a dictionary indexed by OWL objects and the values
        are the corresponding indicies.
        """
        dict_ = {v: k for k, v in enumerate(self._collection)}
        return dict_

    @property
    def as_str(self):
        """Returns the list of entities as string names."""
        return list(self._name_owlobject.keys())

    @property
    def as_owl(self):
        """Returns the list of entities as OWL objects."""
        return list(self._name_owlobject.values())

    @property
    def as_dict(self):
        """Returns the dictionary of entities indexed by their names."""
        return self._name_owlobject

    @property
    def as_index_dict(self):
        """Returns the dictionary of entities indexed by their names."""
        return self._index_dict


class OWLClasses(Entities):
    """
    Iterable for :class:`org.semanticweb.owlapi.model.OWLClass`
    """

    def check_owl_type(self, collection):
        for item in collection:
            if not isinstance(item, OWLClass):
                raise TypeError("Type of elements in collection must be OWLClass.")
        return collection

    def to_str(self, owl_class):
        name = str(owl_class.toStringID())
        return name


class OWLIndividuals(Entities):
    """
    Iterable for :class:`org.semanticweb.owlapi.model.OWLIndividual`
    """

    def check_owl_type(self, collection):
        for item in collection:
            if not isinstance(item, OWLIndividual):
                raise TypeError("Type of elements in collection must be OWLIndividual.")
        return collection

    def to_str(self, owl_individual):
        name = str(owl_individual.toStringID())
        return name


class OWLObjectProperties(Entities):
    """
    Iterable for :class:`org.semanticweb.owlapi.model.OWLObjectProperty`
    """

    def check_owl_type(self, collection):
        for item in collection:
            if not isinstance(item, OWLObjectProperty):
                raise TypeError("Type of elements in collection must be OWLObjectProperty.")
        return collection

    def to_str(self, owl_class):
        name = str(owl_class.toString())
        if name.startswith("<"):
            name = name[1:-1]
        return name
