import os
import sys
sys.path.append("../../../")
import mowl
mowl.init_jvm("3g")
import numpy as np
from scipy.stats import rankdata


from mowl.model import Model
from mowl.reasoning.base import MOWLReasoner
from mowl.corpus.base import extract_axiom_corpus
from jpype.types import *

from org.semanticweb.owlapi.model import AxiomType
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.HermiT import Reasoner

from sklearn.metrics import pairwise_distances

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import logging

MAX_FLOAT = np.finfo(np.float32).max
TEMP_CORPUS_FILE = "temp_corpus_file"

class Onto2Vec(Model):

    '''
    :param dataset: Dataset composed by training, validation and testing sets, each of which are in OWL format.
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param model_outfile: Path to save the final model
    :type model_outfile: str
    :param vector_size: Dimensionality of the word vectors. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type vector_size: int
    :param wv_epochs: Number of epochs for the Word2Vec model
    :type wv_epochs: int
    :param window: Maximum distance between the current and predicted word within a sentence. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type window: int
    :param workers: Number of threads to use for the random walks and the Word2Vec model.
    :type workers: int
    :param corpus_outfile: Path for savings the corpus. If not set the walks will not be saved.
    :type corpus_outfile: str
    '''

    def __init__(
            self,
            dataset,
            model_outfile,
            corpus_outfile = None,
            reasoner = "elk",
            wv_epochs = 10,
            vector_size = 100,
            window = 5,
            workers = 1):
        
        super().__init__(dataset)

        if corpus_outfile is None:
            self.axioms_filepath = TEMP_CORPUS_FILE
        else:
            self.axioms_filepath = corpus_outfile
        self.wv_epochs = wv_epochs
        self.vector_size = vector_size
        self.window = window
        self.model_filepath = model_filepath
        

        self.w2v_model = None
        
        if reasoner == "elk":
            reasoner_factory = ElkReasonerFactory()
            reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
            reasoner.precomputeInferences()
        elif reasoner == "hermit":
            reasoner_factory = Reasoner.ReasonerFactory()
            reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
            reasoner.precomputeInferences()

        self.mowl_reasoner = MOWLReasoner(reasoner)

    def _load_pretrained_model(self):
        return None

    def train(self):
        if not os.path.exists(self.axioms_filepath):
            self.mowl_reasoner.infer_subclass_axioms(self.dataset.ontology)
            self.mowl_reasoner.infer_equiv_class_axioms(self.dataset.ontology)

            extract_axiom_corpus(self.dataset.ontology, self.axioms_filepath)

        sentences = LineSentence(self.axioms_filepath)

        self.w2v_model = self._load_pretrained_model()

        if not self.w2v_model:
            self.w2v_model = Word2Vec(
                sentences=sentences,
                sg=1,
                min_count=1,
                vector_size=self.vector_size,
                window = self.window,
                epochs = self.wv_epochs,
                workers = self.workers)
        else:
            # retrain the pretrained model with our axioms
            self.w2v_model.build_vocab(sentences, update=True)
            self.w2v_model.train(
                sentences,
                total_examples=self.w2v_model.corpus_count,
                sg=1,
                min_count=1,
                vector_size=self.vector_size,
                window = self.window,
                epochs = self.wv_epochs,
                workers = self.workers)
            # (following example from: https://github.com/bio-ontology-research-group/opa2vec/blob/master/runWord2Vec.py )

        self.w2v_model.save(self.model_filepath)


    def train_or_load_model(self):
        if not os.path.exists(self.model_filepath):
            self.train()
        if not self.w2v_model:
            self.w2v_model = gensim.models.Word2Vec.load(
                self.model_filepath)


    def get_classes_pairs_from_axioms(self, data_subset, filter_properties):
        classes_pairs_set = set()
        all_classes_set = set()
        for axiom in data_subset.getAxioms():
            if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
                continue
            try:
                # see Java methods of classes:
                # http://owlcs.github.io/owlapi/apidocs_4/uk/ac/manchester/cs/owl/owlapi/OWLSubClassOfAxiomImpl.html
                # http://owlcs.github.io/owlapi/apidocs_4/uk/ac/manchester/cs/owl/owlapi/OWLObjectSomeValuesFromImpl.html
                cls1 = str(axiom.getSubClass())
                cls2 = str(axiom.getSuperClass().getFiller())
                object_property = str(axiom.getSuperClass().getProperty())
                if object_property in filter_properties:
                    classes_pairs_set.add((cls1, cls2))
                    all_classes_set.add(cls1)
                    all_classes_set.add(cls2)
            except AttributeError as e:
                # no getFiller on some axioms (which are not related to protein-protein interactions, but are other kinds of axioms)
                pass
        return list(all_classes_set), list(classes_pairs_set)


    def evaluate_ppi(self, ppi_axiom_properties=['<http://interacts_with>']):
        """
        Evaluate predicted protein-protein interactions relative to the test ontology, which has the set of interactions kept back from model training.
        """
        self.train_or_load_model()
        model = self.w2v_model
        training_classes, training_classes_pairs = self.get_classes_pairs_from_axioms(self.dataset.ontology, ppi_axiom_properties)
        _, testing_classes_pairs = self.get_classes_pairs_from_axioms(self.dataset.testing, ppi_axiom_properties)

        # some classes in the training set don't make it into the model (maybe their frequency is too low)
        available_training_classes = [c for c in training_classes if c in model.wv]
        class_to_index = {available_training_classes[i]: i for i in range(0, len(available_training_classes))}

        # dict "protein-index-1 => set( protein-indexes-2 )" of the trained PPI pairs
        training_pairs_exclude_indexes = dict()
        for training_pair in training_classes_pairs:
            i1 = class_to_index.get(training_pair[0])
            i2 = class_to_index.get(training_pair[1])
            if i1 is not None and i2 is not None:
                exclude_ids_set = training_pairs_exclude_indexes.get(i1, set())
                training_pairs_exclude_indexes[i1] = exclude_ids_set
                exclude_ids_set.add(i2)

        testing_classes_pairs = sorted(testing_classes_pairs, key=lambda pair: pair[0])
        embeddings = model.wv[available_training_classes]
        observed_ranks = list()
        previous_i1 = None  # to preserve memory, we compare one protein to all the others at a time
        for testing_pair in testing_classes_pairs:
            i1 = class_to_index.get(testing_pair[0])
            i2 = class_to_index.get(testing_pair[1])
            if i1 is not None and i2 is not None:
                # prepare a new row of class comparisons
                if previous_i1 != i1:
                    previous_i1 = i1
                    # Word2Vec.n_similarity only returns an aggregated similarity of all vectors, so staying with this:
                    class_distances = pairwise_distances([embeddings[i1]], embeddings, metric='cosine')[0]

                    # disregard the protein-protein interactions which came naturally from the training set
                    exclude_ids_set = training_pairs_exclude_indexes.get(i1, set())
                    for exclude_i2 in exclude_ids_set:
                        class_distances[exclude_i2] = MAX_FLOAT
                    # disregard the similarity of protein with itself
                    class_distances[i1] = MAX_FLOAT

                    # For each protein, it is ranked how similar (per the model) it is to the current i1.
                    # The lower the rank, the higher the protein similarity.
                    ranked_indexes = rankdata(class_distances, method='average')
                observed_ranks.append(ranked_indexes[i2])

        # We queried the similarity ranks of all the testing set protein-protein interactions, and collected the
        # ranks in observed_ranks. Let's bin the ranks and see if good ranks appear more often, and also
        # calculate the mean rank.
        histogram = np.histogram(observed_ranks, bins=[0, 1.1, 10.1, 100.1, 10000000])[0]
        rank_1 = histogram[0]
        rank_10 = histogram[0] + histogram[1]
        rank_100 = histogram[0] + histogram[1] + histogram[2]
        return(np.mean(observed_ranks), rank_1, rank_10, rank_100)

