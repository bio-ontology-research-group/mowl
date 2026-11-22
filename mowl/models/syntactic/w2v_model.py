"""Word2Vec on the axioms (and annotations) of the ontology written as sentences."""

import logging
import os

from deprecated.sphinx import versionadded

import mowl.error.messages as msg
from mowl.base_models import SyntacticModel
from mowl.models.composed.sequence import ComposedSequenceModel
from mowl.utils.device import default_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@versionadded(version="0.2.0")
class SyntacticPlusW2VModel(ComposedSequenceModel, SyntacticModel):
    """
    Model that combines corpus generation with Word2Vec training.

    It is the combination of parts (:mod:`mowl.representation.sequence`) of the methods
    ``onto2vec`` and ``opa2vec``: the axiom sentences (and, with annotations, the annotation
    sentences after them) written to :attr:`corpus_filepath` by :meth:`generate_corpus`, a
    Word2Vec encoder (:meth:`set_w2v_model`), the vector of every entity's token, the
    similarity :math:`1 - \\sigma(x \\cdot y)` (:attr:`evaluation_model`) and a pair
    classifier for the ranking of candidate axioms.

    An annotation sentence names its class by the IRI, the token of the class in the axiom
    sentences (``annotation_entity_form="bracketed iri"`` of :meth:`generate_corpus` writes
    ``<IRI>`` as before). A Word2Vec model that has a vocabulary keeps its vectors when it is
    trained again; pass ``restart=True`` to :meth:`set_w2v_model` to start anew on every
    :meth:`train`.

    :param corpus_filepath: the filepath where the corpus is saved. If None, the corpus file
        is saved in a temporary file.
    :type corpus_filepath: str
    """

    # Word2Vec gives one vector per class; an asymmetric axiom score needs a scoring head
    # (mowl.evaluation.model_scoring.PairClassifierHead): the scorer of the geometry.
    axiom_scoring = "vectors"
    scored_forms = ("gci0", "gci2")

    def __init__(self, dataset, model_filepath=None, corpus_filepath=None):
        from mowl.representation.library.walks_views import CorpusFile
        super().__init__(dataset, model_filepath=model_filepath, device=default_device(),
                         source=CorpusFile(None),
                         _base_options={"corpus_filepath": corpus_filepath})
        self._is_pretrained = False
        self.module.source._path = self.corpus_filepath
        self._annotation_entity_form = "iri"

    def _corpus_source(self, with_annotations, annotation_entity_form):
        from mowl.representation.library.walks_views import (AxiomCorpus, AnnotationCorpus,
                                                             Sentences, Mixed)
        axioms = Sentences(AxiomCorpus(), path=self.corpus_filepath)
        if not with_annotations:
            return axioms
        axioms = Sentences(AxiomCorpus())
        annotations = Sentences(AnnotationCorpus(entity_form=annotation_entity_form))
        return Mixed(axioms, annotations, path=self.corpus_filepath)

    def generate_corpus(self, save=True, with_annotations=False,
                        annotation_entity_form="iri"):
        """Generates the corpus of the training ontology. It uses the Manchester OWL Syntax.

        :param save: if True, the corpus is saved into the model filepath, otherwise, the
            corpus is returned as a list of sentences. Default is True.
        :type save: bool, optional
        :param with_annotations: if True, the corpus is generated with the annotations,
            otherwise, the corpus is generated only with the axioms. Default is False.
        :type with_annotations: bool, optional
        :param annotation_entity_form: ``"iri"`` (default) or ``"bracketed iri"``, see
            :class:`~mowl.representation.library.walks_views.AnnotationCorpus`.
        """
        source = self._corpus_source(with_annotations, annotation_entity_form)
        if not save:
            sources = source.sources if hasattr(source, "sources") else [source]
            corpus = []
            for s in sources:
                corpus += s.sentences(self.dataset)
            return corpus
        self.module.source = source
        source.write(self.dataset)
        self._written = True
        logger.info(f"Corpus saved in {self.corpus_filepath}")
        self._save_corpus = save
        self._with_annotations = with_annotations
        self._annotation_entity_form = annotation_entity_form

    def _ensure_sequences(self):
        if not os.path.exists(self.corpus_filepath):
            raise FileNotFoundError(msg.CORPUS_NOT_GENERATED)
        if self.axioms_added:
            self.module.source.write(self.dataset)
            self.axioms_added = False
        self._written = True

    def train(self, epochs=None):
        """
        Triggers the Word2Vec training process.

        :param epochs: Number of epochs to train the model. If None, the value of the epochs
            parameter passed to the constructor will be used.
        :type epochs: int
        """
        super().train(epochs)

    def add_axioms(self, *axioms):
        """Adds the axioms to the dataset and writes the corpus anew (with the settings of
        the last :meth:`generate_corpus`)."""
        from mowl.representation.library.walks_views import CorpusFile
        if isinstance(self.module.source, CorpusFile):
            self.module.source = self._corpus_source(self._with_annotations,
                                                     self._annotation_entity_form)
        super().add_axioms(*axioms)
