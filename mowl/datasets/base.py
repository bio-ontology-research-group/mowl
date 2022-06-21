import numpy as np
import tarfile
import pathlib
import os
from typing import Optional
from jpype import *
import jpype.imports
import requests


# OWLAPI imports
from org.semanticweb.owlapi.model import OWLOntology
from org.semanticweb.owlapi.model import OWLOntologyLoaderConfiguration
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.reasoner import ConsoleProgressMonitor
from org.semanticweb.owlapi.reasoner import SimpleConfiguration
from org.semanticweb.owlapi.reasoner import InferenceType
from org.semanticweb.owlapi.util import InferredClassAssertionAxiomGenerator
from org.semanticweb.owlapi.util import InferredEquivalentClassAxiomGenerator
from org.semanticweb.owlapi.util import InferredSubClassAxiomGenerator
from org.semanticweb.elk.owlapi import ElkReasonerFactory

from mowl.projection.taxonomyRels.model import TaxonomyWithRelsProjector
class Dataset(object):

    """This class provide training, validation and testing datasets encoded as OWL ontologies.

    :param ontology: Training dataset
    :type ontology: org.semanticweb.owlapi.model.OWLOntology
    :param validation: Validation dataset
    :type validation: org.semanticweb.owlapi.model.OWLOntology
    :param testing: Testing dataset
    :type testing: org.semanticweb.owlapi.model.OWLOntology
    """
    
    ontology: OWLOntology
    validation: OWLOntology
    testing: OWLOntology

class PathDataset(Dataset):
    """Loads the dataset from ontology documents.

    :param ontology_path: Training dataset
    :type ontology_path: str
    :param validation_path: Validation dataset
    :type validation_path: str
    :param testing_path: Testing dataset
    :type testing_path: str

    """

    ontology_path: str
    validation_path: str
    testing_path: str
    _ontology: OWLOntology
    _validation: OWLOntology
    _testing: OWLOntology
    
    def __init__(self, ontology_path: str, validation_path: str, testing_path: str):
        self.ontology_path = ontology_path
        self.validation_path = validation_path
        self.testing_path = testing_path
        self.ont_manager = OWLManager.createOWLOntologyManager()
        self.data_factory = self.ont_manager.getOWLDataFactory()
        self.reasoner = None
        self._loaded = False
        
    @property
    def ontology(self):
        if not self._loaded:
            self._load()
        return self._ontology


    @property
    def validation(self):
        if not self._loaded:
            self._load()
        return self._validation

    @property
    def testing(self):
        if not self._loaded:
            self._load()
        return self._testing

    def _load(self):

        self._ontology = self.ont_manager.loadOntologyFromOntologyDocument(
            java.io.File(self.ontology_path))
        if not self.validation_path is None:
            self._validation =  self.ont_manager.loadOntologyFromOntologyDocument(
                java.io.File(self.validation_path))
        else:
            self._validation = None

        if not self.testing_path is None:
            self._testing =  self.ont_manager.loadOntologyFromOntologyDocument(
                java.io.File(self.testing_path))
        else:
            self._testing = None
        self._loaded = True

    def _create_reasoner(self):
        progressMonitor = ConsoleProgressMonitor()
        config = SimpleConfiguration(progressMonitor)
        reasoner_factory = ElkReasonerFactory()
        self.reasoner = reasoner_factory.createReasoner(self.ontology, config)
        self.reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)


    def infer_axioms(self):
        print("The infer_axioms method is deprecated and will be removed soon. Please consider using the methods existing in the mowl.reasoning module.")
        if not self.reasoner:
            self._create_reasoner()
        assertion_axioms = InferredClassAssertionAxiomGenerator().createAxioms(
            self.data_factory, self.reasoner)
        self.ont_manager.addAxioms(self.ontology, assertion_axioms)
        equivalent_axioms = InferredEquivalentClassAxiomGenerator().createAxioms(
            self.data_factory, self.reasoner)
        self.ont_manager.addAxioms(self.ontology, equivalent_axioms)
        subclass_axioms = InferredSubClassAxiomGenerator().createAxioms(
            self.data_factory, self.reasoner)
        self.ont_manager.addAxioms(self.ontology, subclass_axioms)

    def get_evaluation_classes(self):
        return self.ontology.getClassesInSignature()


    def get_labels(self):
        projector = TaxonomyWithRelsProjector(relations = ["http://has_label"])
        edges = projector.project(self.ontology)
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
        super().__init__(
            os.path.join(dataset_root, 'ontology.owl'),
            os.path.join(dataset_root, 'valid.owl'),
            os.path.join(dataset_root, 'test.owl'))
        self._extract()
        

    def _extract(self):
        ontology_exists = os.path.exists(self.ontology_path)
        validation_exists = os.path.exists(self.validation_path)
        testing_exists = os.path.exists(self.testing_path)
        if ontology_exists and validation_exists and testing_exists:
            return
        with tarfile.open(self.tarfile_path) as tf:
            tf.extractall(path=self.data_root)
        
        
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
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filepath
