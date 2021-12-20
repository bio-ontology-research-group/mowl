
from mowl.model import Model
from mowl.graph.util import parser_factory
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
    :param vector_size: Dimensionality of the word vectors. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type vector_size: int
    :param window: Maximum distance between the current and predicted word within a sentence. Same as :class:`gensim.models.word2vec.Word2Vec`
    :type window: int
    :param num_procs: Number of threads to use for the random walks and the Word2Vec model.
    :type num_procs: int
    '''

    
    def __init__(self, dataset, outfile, bidirectional_taxonomy=False, walk_length = 30, num_walks = 100, vector_size = 100, window = 5, num_procs = 1):

        super().__init__(dataset)

        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.num_procs = num_procs
        self.vector_size = vector_size
        self.window = window
        self.outfile = outfile
        
        self.parserTrain = parser_factory("dl2vec", self.dataset.ontology, bidirectional_taxonomy)
        self.parserTest = parser_factory("dl2vec", self.dataset.testing, bidirectional_taxonomy)

        self.lock = Lock()

    def train(self):

        edges = self.parserTrain.parse()
        entities, _ = Edge.getEntitiesAndRelations(edges)
        entities = list(entities)

        G = nx.Graph()

        for edge in edges:
            src, rel, dst = edge.src(), edge.rel(), edge.dst()
            
            G.add_edge(src, dst)
            G.edges[src,dst]["type"] = rel
            G.nodes[src]["val"] = False
            G.nodes[dst]["val"] = False


        self.run_walk(entities, G)

        logging.info("Walks generated")
        logging.info("Starting to train the Word2Vec model")

        sentences = gensim.models.word2vec.LineSentence("tmp_walks.txt")
        model = gensim.models.Word2Vec(sentences, sg=1, min_count=1, vector_size=self.vector_size, window = self.window, epochs = self.num_walks, workers = self.num_procs)

        model.save(self.outfile)

        
    def run_random_walks(self, G, nodes):

        walks = []
        for count, node in enumerate(nodes):
            
            if G.degree(node) == 0:
                continue

            for i in range(self.num_walks):
                curr_node = node
                
                walk_accumulate=[]

                for j in range(self.walk_length):
                    next_node = random.choice(list(G.neighbors(curr_node)))

                    type_nodes = G.edges[curr_node, next_node]["type"]

                    if curr_node == node:
                        walk_accumulate.append(curr_node)
                    walk_accumulate.append(type_nodes)
                    walk_accumulate.append(next_node)

                    curr_node = next_node

                walks.append(walk_accumulate)
            if count % 1000 == 0:
                print("Done walks for", count, "nodes")

        self.write_file(walks)


    def run_walk(self, nodes,G):
        length = len(nodes) // self.num_procs

        processes = [mp.Process(target=self.run_random_walks, args=(G, nodes[(index) * length:(index + 1) * length])) for index in range(self.num_procs-1)]
        processes.append(mp.Process(target=self.run_random_walks, args=(G, nodes[(self.num_procs-1) * length:len(nodes) - 1])))

        for p in processes:
            p.start()
        for p in processes:
            p.join()
        


    
    def write_file(self, pair):
        with self.lock:
            with open("tmp_walks.txt", "a") as fp:
                for p in pair:
                    for sub_p in p:
                        fp.write(str(sub_p)+" ")
                    fp.write("\n")
