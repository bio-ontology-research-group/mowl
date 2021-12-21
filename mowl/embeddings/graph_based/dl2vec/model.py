
from mowl.model import Model
from mowl.graph.util import parser_factory
from mowl.walking.util import walking_factory
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

class DL2Vec(Model):

    '''
    :param dataset: Dataset in OWL format corresponding
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param outfile: Path to save the final model
    :type outfile: str
    :param bidirectional_taxonomy: If true, the ontology projection into a graph will add inverse edges per each subclass axiom
    :type bidirectional_taxonomy: bool
    :param walk_length: Length of the walk performed for each node
    :type walk_length: int
    :param num_walks: Number of walks performed per node
    :type num_walks: int
    :param alpha: Probability of restart in the walking phase
    :type alpha: float
    :param vector_size: Dimensionality of the word vectors. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type vector_size: int
    :param window: Maximum distance between the current and predicted word within a sentence. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type window: int
    :param num_procs: Number of threads to use for the random walks and the Word2Vec model.
    :type num_procs: int
    '''

    
    def __init__(self, dataset, outfile, bidirectional_taxonomy=False, walking_method = "deepwalk", walk_length = 30, alpha = 0, num_walks = 100, vector_size = 100, window = 5, num_procs = 1):

        super().__init__(dataset)

        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.alpha = alpha
        self.num_procs = num_procs
        self.vector_size = vector_size
        self.window = window
        self.outfile = outfile
        self.walking_method = walking_method
        self.parserTrain = parser_factory("dl2vec", self.dataset.ontology, bidirectional_taxonomy)
        self.parserTest = parser_factory("dl2vec", self.dataset.testing, bidirectional_taxonomy)

        self.lock = Lock()

    def train(self):

        logging.info("Generating graph from ontology...")
        edges = self.parserTrain.parse()
        entities, _ = Edge.getEntitiesAndRelations(edges)
        entities = list(entities)
        logging.info("Finished graph generation")

        logging.info("Generating random walks...")
        walks_outfile = "data/walks.txt"
        walker = walking_factory(self.walking_method, edges, self.num_walks, self.walk_length, self.alpha, num_workers = self.num_procs, outfile=walks_outfile)
        walker.walk()
        logging.info("Walks generated")

        logging.info("Starting to train the Word2Vec model")

        sentences = gensim.models.word2vec.LineSentence(walks_outfile)
        model = gensim.models.Word2Vec(sentences, sg=1, min_count=1, vector_size=self.vector_size, window = self.window, epochs = self.num_walks, workers = self.num_procs)
        logging.info("Word2Vec training finished")
        logging.info(f"Saving model at {self.outfile}")
        model.save(self.outfile)
        logging.info("Model saved")
        