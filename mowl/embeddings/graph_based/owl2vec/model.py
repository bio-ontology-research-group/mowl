from mowl.base_models.model import Model
from mowl.graph.factory import parser_factory
from mowl.walking.factory import walking_factory
from mowl.graph.edge import Edge

import numpy as np
import random
import json
import sys
import os
import gensim
import tempfile
import networkx as nx
from networkx.readwrite import json_graph
import multiprocessing as mp
from threading import Lock
import pickle as pkl
import logging
from threading import Lock

logging.basicConfig(level=logging.INFO)

class OWL2VecStar(Model):
    '''
    :param dataset: Dataset composed by training, validation and testing sets, each of which are in OWL format.
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param outfile: Path to save the final model
    :type outfile: str
    :param bidirectional_taxonomy: If true, the ontology projection into a graph will add inverse edges per each subclass axiom
    :type bidirectional_taxonomy: bool
    :param only_taxonomy: If true, the ontology projection will consider only subclass axioms.
    :type only_taxonomy: bool
    :param include_literals: If true, the graph will also include triples involving data property assertions and annotations.
    :type include_literals: bool
    :param walking_method: Method for generating the walks. Choices are: deepwalk (default), node2vec, walkrdfowl.
    :type walking_method: str
    :param walk_length: Length of the walk performed for each node
    :type walk_length: int
    :param num_walks: Number of walks performed per node
    :type num_walks: int
    :param alpha: Probability of restart in the walking phase. Applicable with DeepWalk
    :type alpha: float
    :param p: Return hyperparameter. Default is 1. Applicable with Node2Vec
    :type p: float
    :param q: In-out hyperparameter. Default is 1. Applicable with Node2Vec.
    :type q: float
    :param vector_size: Dimensionality of the word vectors. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type vector_size: int
    :param wv_epochs: Number of epochs for the Word2Vec model
    :type wv_epochs: int
    :param window: Maximum distance between the current and predicted word within a sentence. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type window: int
    :param workers: Number of threads to use for the random walks and the Word2Vec model.
    :type workers: int
    :param walks_outfile: Path for savings the walks. If not set the walks will not be saved.
    :type walks_outfile: str

    '''
    
    def __init__(self, dataset, outfile, bidirectional_taxonomy=False, include_literals = False, only_taxonomy = False, walking_method = "deepwalk", walk_length = 30, wv_epochs = 10, alpha = 0, num_walks = 100, vector_size = 100, window = 5, workers = 1, p = 1, q=1, walks_outfile = None):

        super().__init__(dataset)

        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.include_literals = include_literals
        self.only_taxonomy = only_taxonomy
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.alpha = alpha
        self.p = p
        self.q = q
        self.workers = workers
        self.wv_epochs = wv_epochs
        self.vector_size = vector_size
        self.window = window
        self.outfile = outfile
        self.walking_method = walking_method
        self.parserTrain = parser_factory("owl2vec_star", self.dataset.ontology, bidirectional_taxonomy = self.bidirectional_taxonomy, include_literals = self.include_literals, only_taxonomy = self.only_taxonomy)
        self.parserTest = parser_factory("owl2vec_star", self.dataset.testing, bidirectional_taxonomy = self.bidirectional_taxonomy, include_literals = self.include_literals, only_taxonomy = self.only_taxonomy)

        self.walks_outfile = walks_outfile
        self.lock = Lock()

    def train(self):
        
        save_walks = True

        logging.info("Generating graph from ontology...")
        edges = self.parserTrain.parse()
        entities, _ = Edge.getEntitiesAndRelations(edges)
        self.entities = list(entities)
        logging.info("Finished graph generation")

        logging.info("Generating random walks...")

        if self.walks_outfile is None:
            save_walks = False
            self.walks_outfile = "walks_temporary_output_file.tmp"

        walker = walking_factory(self.walking_method, edges, self.num_walks, self.walk_length, self.walks_outfile, workers = self.workers, alpha = self.alpha, p = self.p, q= self.q)

        walker.walk()

        logging.info("Walks generated")
        if save_walks:
            logging.info(f"Walks saved at {self.walks_outfile}")

        logging.info("Starting to train the Word2Vec model")

        sentences = gensim.models.word2vec.LineSentence(self.walks_outfile)

        model = gensim.models.Word2Vec(sentences, sg=1, min_count=1, vector_size=self.vector_size, window = self.window, epochs = self.wv_epochs, workers = self.workers)
        logging.info("Word2Vec training finished")
        logging.info(f"Saving model at {self.outfile}")

        if not save_walks:
            os.remove(self.walks_outfile)
        model.save(self.outfile)
        logging.info("Model saved")
        
