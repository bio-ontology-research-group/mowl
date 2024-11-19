from mowl.base_models.model import Model
import mowl.error.messages as msg
from mowl.corpus import extract_annotation_corpus, extract_and_save_annotation_corpus, extract_axiom_corpus, extract_and_save_axiom_corpus
import tempfile

from deprecated.sphinx import versionadded


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SyntacticModel(Model):
    """
    Base class for syntactic methods. By *syntactic*, we mean methods that use the syntax of the ontology to generate the corpus.

    :param corpus_filepath: the filepath where the corpus is saved. If None, the corpus file is saved in a temporary file.
    :type corpus_filepath: str
    """

    
    def __init__(self, *args, corpus_filepath=None, **kwargs):
        super().__init__(*args, **kwargs)

        if corpus_filepath is not None and not isinstance(corpus_filepath, str):
            raise TypeError("Optional parameter 'corpus_filepath' must be of type str.")

        self._corpus_filepath = corpus_filepath
        self._corpus = None
        self._save_corpus = True
        self._with_annotations = False

    @property
    def corpus_filepath(self):
        """Path for saving the corpus.

        :rtype: str
        """
        if self._corpus_filepath is None:
            corpus_filepath = tempfile.NamedTemporaryFile()
            self._corpus_filepath = corpus_filepath.name
        return self._corpus_filepath

        
    @property
    def corpus(self):
        """
        Corpus generated from the ontology.

        :rtype: list
        """
        if self._corpus is None:
            raise AttributeError(msg.CORPUS_NOT_GENERATED)
    
        return self._corpus

    def generate_corpus(self, save = True, with_annotations=False):
        """Generates the corpus of the training ontology. It uses the Manchester OWL Syntax.
        
        :param save: if True, the corpus is saved into the model filepath, otherwise, the corpus is returned as a list of sentences. Default is True.
        :type save: bool, optional
        :param with_annotations: if True, the corpus is generated with the annotations, otherwise, the corpus is generated only with the axioms. Default is False.
        :type with_annotations: bool, optional
        """
        if save:
            extract_and_save_axiom_corpus(self.dataset.ontology,
                                               self.corpus_filepath,
                                               mode="w")
            if with_annotations:
                extract_and_save_annotation_corpus(self.dataset.ontology,
                                                   self.corpus_filepath,
                                                   mode="a")
        else:
            corpus = extract_axiom_corpus(self.dataset.ontology)
            if with_annotations:
                corpus += extract_annotation_corpus(self.dataset.ontology)

        if not save:
            return corpus
        else:
            logger.info(f"Corpus saved in {self.corpus_filepath}")

        self._save_corpus = save
        self._with_annotations = with_annotations


    @versionadded(version="0.4.0")
    def load_corpus(self):
        """Loads the corpus from the corpus filepath.

        :rtype: list
        """
        if self._corpus is None:
            with open(self.corpus_filepath, "r") as f:
                corpus = f.readlines()
                corpus = [sentence.strip() for sentence in corpus]
                self._corpus = corpus
                
        return self._corpus
        
    def train(self):
        raise NotImplementedError
