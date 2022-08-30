"""
This module contains classes intended to deal with mOWL datasets.
"""

import tarfile
import pathlib
import os

from jpype import java
import requests

# OWLAPI imports
from org.semanticweb.owlapi.model import OWLOntology, OWLClass, OWLObjectProperty
from org.semanticweb.owlapi.apibinding import OWLManager

from mowl.projection.taxonomyRels.model import TaxonomyWithRelsProjector
from mowl.owlapi.adapter import OWLAPIAdapter
from mowl.owlapi.defaults import TOP, BOT

class PathDataset():
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

        #Checks on training file path
        if not isinstance(ontology_path, str):
            raise TypeError("Training ontology path must be a string.")

        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Training ontology file not found {ontology_path}")

        #Checks on validation file path
        if validation_path is not None:
            if not isinstance(validation_path, str):
                raise TypeError("Training validation path must be a string.")

            if not os.path.exists(validation_path):
                raise FileNotFoundError(f"Validation ontology file not found {validation_path}")

        #Checks on testing file path
        if testing_path is not None:
            if not isinstance(testing_path, str):
                raise TypeError("Training testing path must be a string.")

            if not os.path.exists(testing_path):
                raise FileNotFoundError(f"Testing ontology file not found {testing_path}")

        self.ontology_path = ontology_path
        self.validation_path = validation_path
        self.testing_path = testing_path
        self._loaded = False
        self._classes = None
        self._object_properties = None
        self._evaluation_classes = None


    def _load(self):
        if self._loaded:
            return

        ont_manager = OWLManager.createOWLOntologyManager()
        self._ontology = ont_manager.loadOntologyFromOntologyDocument(
            java.io.File(self.ontology_path))
        if not self.validation_path is None:
            self._validation = ont_manager.loadOntologyFromOntologyDocument(
                java.io.File(self.validation_path))
        else:
            self._validation = None

        if not self.testing_path is None:
            self._testing = ont_manager.loadOntologyFromOntologyDocument(
                java.io.File(self.testing_path))
        else:
            self._testing = None
        self._loaded = True

    @property
    def ontology(self):
        """Training dataset

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        self._load()
        return self._ontology

    @property
    def validation(self):
        """Validation dataset

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        self._load()
        return self._validation

    @property
    def testing(self):
        """Testing ontology

        :rtype: org.semanticweb.owlapi.model.OWLOntology
        """
        self._load()
        return self._testing

    @property
    def classes(self):
        """List of classes in the dataset. The classes are collected from training, validation and
        testing ontologies using the OWLAPI method ``ontology.getClassesInSignature()``.

        :rtype: OWLClasses
        """
        self._load()
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
    def object_properties(self):
        """List of object properties (relations) in the dataset. The object
        properties are collected from training, validation and testing
        ontologies using the OWLAPI
        method ``ontology.getObjectPropertiesInSignature()``.

        :rtype: OWLObjectProperties
        """
        self._load()
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
    def evaluation_classes(self):
        """List of classes used for evaluation. Depending on the dataset, this
        method could return a single :class:`OWLClasses` object
        (as in :class:`PPIYeastDataset <mowl.datasets.builtin.PPIYeastDataset>`)
        or a tuple of :class:`OWLClasses` objects (as in :class:`GDAHumanDataset <mowl.datasets.builtin.GDAHumanDataset>`). If not overriden, this method returns the classes in the testing ontology obtained from the OWLAPI method ``getClassesInSignature()`` as a :class:`OWLClasses` object.
        """
        self._load()
        if self._evaluation_classes is None:
            classes = self._testing.getClassesInSignature()
            self._evaluation_classes = OWLClasses(classes)
        return self._evaluation_classes

    @property
    def labels(self):
        """This method returns labels of entities as a dictionary. To be
        called, the training ontology must contain axioms of the form
        :math:`class_1 \sqsubseteq \exists http://has\_label . class_2`.

        :rtype: dict
        """
        self._load()
        projector = TaxonomyWithRelsProjector(relations = ["http://has_label"])
        edges = projector.project(self._ontology)
        labels = {str(e.src()): str(e.dst()) for e in edges}
        return labels

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

        ontology_path = os.path.join(dataset_root, 'ontology.owl')
        validation_path = os.path.join(dataset_root, 'valid.owl')
        testing_path = os.path.join(dataset_root, 'test.owl')

        ontology_exists = os.path.exists(ontology_path)
        validation_exists = os.path.exists(validation_path)
        testing_exists = os.path.exists(testing_path)
        if not (ontology_exists and validation_exists and testing_exists):
            self._extract()

        super().__init__(
            ontology_path,
            validation_path,
            testing_path)

    def _extract(self):
        with tarfile.open(self.tarfile_path) as tarf:
            tarf.extractall(path=self.data_root)

class RemoteDataset(TarFileDataset):
    """Loads the dataset from a remote URL.

    :param url: URL location of the dataset
    :type url: str
    :param data_root: Root directory
    :type data_root: str
    """

    url: str
    data_root: str

    def __init__(self, url: str, data_root='./'):
        self.url = url
        self.data_root = data_root
        tarfile_path = self._download()
        super().__init__(tarfile_path)

    def _download(self):
        filename = self.url.split('/')[-1]
        filepath = os.path.join(self.data_root, filename)
        if os.path.exists(filepath):
            return filepath
        with requests.get(self.url, stream=True) as req:
            req.raise_for_status()
            with open(filepath, 'wb') as writer:
                for chunk in req.iter_content(chunk_size=8192):
                    writer.write(chunk)
        return filepath


class Entities():
    """Abstract class containing OWLEntities indexed by they IRIs"""
    def __init__(self, collection):
        self._collection = self.check_owl_type(collection)
        self._name_owlobject = self.to_dict()

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
        dict_ = {}
        for ent in self._collection:
            name = self.to_str(ent)
            if not name in dict_:
                dict_[name] = ent
        dict_ = dict(sorted(dict_.items()))
        return dict_

    @property
    def as_str(self):
        """Returns the list of entities as string names."""
        return list(self._name_owlobject.keys())

    @property
    def as_owl(self):
        """Returns the list of entities as OWL objects."""
        return list(self._name_owlobject.values())

class OWLClasses(Entities):
    """Class containing OWL classes indexed by they IRIs"""
    def check_owl_type(self, collection):
        for item in collection:
            if not isinstance(item, OWLClass):
                raise TypeError("Type of elements in collection must be OWLClass.")
        return collection

    def to_str(self, owl_class):
        name = str(owl_class.toStringID())
        return name

class OWLObjectProperties(Entities):
    """Class containing OWL object properties indexed by they IRIs"""
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