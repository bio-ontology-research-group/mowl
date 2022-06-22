import os
import sys
sys.path.append("../../../")
import mowl
mowl.init_jvm("3g")

from mowl.embeddings.onto2vec.model import Onto2Vec
from jpype import java
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import OWLOntology, OWLLiteral
from org.semanticweb.owlapi.search import EntitySearcher
import gensim


class OPA2Vec(Onto2Vec):
    annotations_ontology: OWLOntology

    def __init__(self, dataset, pretrained_w2v_model_filepath=None, w2v_params={}):
        """
        Ontologies Plus Annotations to Vectors: OPA2Vec

        Based on Onto2Vec, but adding information from class annotations from an OWL ontology.

        :param dataset: MOWL dataset to use for training and testing the model
        :param owl_filename: The OWL file which classes' annotations are added to the generated axioms
        :param w2v_params: optional word2vec parameters
        """
        super().__init__(dataset, w2v_params)
        # Currently, models store their files next to the dataset files. Override the Onto2Vec paths
        # so that OPA2Vec can use its own axioms and word2vec model.
        self.axioms_filepath = os.path.join(
            dataset.data_root, dataset.dataset_name, 'opa2vec_axioms.o2v')
        self.model_filepath = os.path.join(
            dataset.data_root, dataset.dataset_name, 'opa2vec_w2v.model')

        self.pretrained_w2v_model_filepath = pretrained_w2v_model_filepath
        self.ont_manager = OWLManager.createOWLOntologyManager()
        self.annotations_ontology = dataset.ontology

    def _create_axioms_corpus(self):
        super()._create_axioms_corpus()
        with open(self.axioms_filepath, 'a') as f:
            for owl_class in self.annotations_ontology.getClassesInSignature():
                cls = str(owl_class)
                if str(owl_class) == '<http://purl.obolibrary.org/obo/GO_0005740>':
                    pass  # useful breakpoint
                annotations = EntitySearcher.getAnnotations(owl_class, self.annotations_ontology)
                for annotation in annotations:
                    if isinstance(annotation.getValue(), OWLLiteral):
                        property = str(annotation.getProperty()).replace("\n", " ")
                        # could filter on property
                        value = str(annotation.getValue().getLiteral()).replace("\n", " ")
                        f.write(f'{cls} {property} {value}\n')

    def _load_pretrained_model(self):
        if self.pretrained_w2v_model_filepath:
            return gensim.models.Word2Vec.load(self.pretrained_w2v_model_filepath)
        return None
