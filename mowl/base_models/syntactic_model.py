from mowl.base_models.model import Model
import mowl.error.messages as msg
from mowl.corpus import extract_annotation_corpus, extract_and_save_annotation_corpus, extract_axiom_corpus, extract_and_save_axiom_corpus
import tempfile

class SyntacticModel(Model):

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
        """Path for saving the model.

        :rtype: str
        """
        if self._corpus_filepath is None:
            corpus_filepath = tempfile.NamedTemporaryFile()
            self._corpus_filepath = corpus_filepath.name
        return self._corpus_filepath

        
    @property
    def corpus(self):
        if self._corpus is None:
            raise AttributeError(msg.CORPUS_NOT_GENERATED)
    
        return self._corpus


    def generate_corpus(self, save = True, with_annotations=False):
        """Generates the corpus of the training ontology. It uses the Manchester OWL Syntax.
        
        :param save: if True, the corpus is saved into the model filepath, otherwise, the corpus is returned as a list of sentences. Default is True.
        :type save: bool
        :param with_annotations: if True, the corpus is generated with the annotations, otherwise, the corpus is generated only with the axioms. Default is False.
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
            print(f"Corpus saved in {self.corpus_filepath}")

        self._save_corpus = save
        self._with_annotations = with_annotations


    def train(self):
        raise NotImplementedError
