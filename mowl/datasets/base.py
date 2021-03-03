import numpy as np
import tarfile
import pathlib

# OWLAPI imports
from org.semanticweb.owlapi.model import OWLOntology
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.reasoner import ConsoleProgressMonitor
from org.semanticweb.owlapi.reasoner import SimpleConfiguration
from org.semanticweb.owlapi.reasoner import InferenceType
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.util import InferredClassAssertionAxiomGenerator
from org.semanticweb.owlapi.util import InferredEquivalentClassAxiomGenerator
from org.semanticweb.owlapi.util import InferredSubClassAxiomGenerator


Triples = np.ndarray

def load_triples(filepath, delimiter='\t', encoding=None):
    return np.loadtxt(
        fname=filepath,
        dtype=str,
        comments='@Comment@ Head Relation Tail',
        delimiter=delimiter,
        encoding=encoding,
    )


class Dataset(object):

    ontology: OWLOntology
    validation: Triples
    testing: Triples



class PathDataset(Dataset):

    ontology_path: str
    validation_path: str
    testing_path: str
    _ontology: OWLOntology
    _validation: Triples
    _testing: Triples
    
    def __init__(self, ontology_path: str, validation_path: str, testing_path: str):
        self.ontology_path = ontology_path
        self.validation_path = validation_path
        self.testing_path = testing_path
        self.ont_manager = OWLManager.createOWLOntologyManager()
        self.data_factory = self.ont_manager.getOWLDataFactory()
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
        self._validation = load_triples(self.validation_path)
        self._testing = load_triples(self.testing_path)

    def _create_reasoner(self):
        progressMonitor = ConsoleProgressMonitor()
        config = SimpleConfiguration(progressMonitor)
        reasoner_factory = ElkReasonerFactory()
        self.reasoner = reasoner_factory.createReasoner(self.ontology, config)
        self.reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

    def infer_axioms(self):
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
        

class TarFileDataset(PathDataset):
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
            os.path.join(dataset_root, 'valid.tsv'),
            os.path.join(dataset_root, 'test.tsv'))
        self._extract()
        

    def _extract(self):
        ontology_exists = os.path.exists(self.ontology_path)
        validation_exists = os.path.exists(self.validation_path)
        testing_exists = os.path.exists(self.testing_path)
        if ontology_exists and validation_exists and testing_exists:
            return
        with tarfile.open(fileobj=self.tarfile_path) as tf:
            tf.extractall(path=self.data_root)
        
        
class RemoteDataset(TarFileDataset):

    url: str
    data_root: str
    
    def __init__(self, url: str, data_root='data/'):
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
