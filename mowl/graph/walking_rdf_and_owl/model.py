from mowl.model import Model

import os
import time
import numpy as np
import gzip
import subprocess
import multiprocessing as mp
from functools import reduce
import operator
from scipy.stats import rankdata
from collections import Counter
from functools import partial

import jpype
import jpype.imports

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

from java.lang import Runnable, Thread, Object
from java.io import PipedInputStream, PipedOutputStream, File

from org.semanticweb.owlapi.model import IRI
from org.apache.jena.rdf.model import ModelFactory
from org.apache.jena.util import FileManager

from java.util import HashMap, ArrayList
from java.util.concurrent import ExecutorService, Executors

from org.mowl.WRO import WorkerThread, WROEval, GenPred

class WalkRdfOwl(Model):
    def __init__(self, 
                    dataset, 
                    corpus_file_path, 
                    embeddings_file_path, 
                    number_walks, 
                    length_walk, 
                    embedding_size, 
                    window, 
                    min_count,
                    undirected=False,
                    data_root = "."
):

        super().__init__(dataset)    
        self.data_root = data_root
        self.corpus_file_path = f"{self.data_root}/{corpus_file_path}"
        self.embeddings_file_path = f"{self.data_root}/{embeddings_file_path}"
        self.number_walks = number_walks
        self.length_walk = length_walk
        self.undirected = undirected
        # Skip-gram params
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        

    def gen_graph(self, format= "RDF/XML"):

        tmp_data_file = File.createTempFile(f"{self.data_root}/temp_file", '.tmp')
    

        #self.dataset.infer_axioms()
        self.dataset.ont_manager.saveOntology(self.dataset.ontology, IRI.create(tmp_data_file.toURI()))


        filename = tmp_data_file.toURI()
        model = ModelFactory.createDefaultModel()
        infile = FileManager.get().open(filename.toString())


        model.read(infile, None, format)        

        edge_list = HashMap()

        
        count = 0
        print("Generating graph...")
        for stmt in model.listStatements():
           
            pred = stmt.getPredicate()
            subj = stmt.getSubject()
            obj =  stmt.getObject()

            if (subj.isURIResource() and obj.isURIResource()):
                pred = str(pred)
                subj = str(subj)
                obj =  str(obj)

                neighbor = ArrayList()
                neighbor.add(pred)
                neighbor.add(obj)

                if edge_list.containsKey(subj):
                    edge_list.get(subj).add(neighbor)
        
                else:
                    neighbors = ArrayList()
                    neighbors.add(neighbor)
                    edge_list.put(subj, ArrayList(neighbors))
                
                if (self.undirected):
                    neighbor = ArrayList()
                    neighbor.add(pred)
                    neighbor.add(subj)

                    if edge_list.containsKey(obj):
                        edge_list.get(obj).add(neighbor)
            
                    else:
                        neighbors = ArrayList()
                        neighbors.add(neighbor)
                        edge_list.put(obj, ArrayList(neighbors))
         
        return edge_list


    def generate_corpus_and_embeddings(self, graph):
        
        print("Started walking...")
        start = time.time()
        n_cores = os.cpu_count()

        executor =  Executors.newFixedThreadPool(n_cores)

        #Just to clear the file before writing again on it.
        f = open(self.corpus_file_path, 'w')
        f.close()

        sources = ArrayList(graph.keySet())
        
        out_file_j = jpype.JObject(self.corpus_file_path, jpype.JClass("java.lang.String"))
        
        with jpype.synchronized(graph):
            for i in range(len(sources)):
                worker = WorkerThread(out_file_j, graph, self.number_walks, self.length_walk, sources[i])
                executor.execute(worker)
                
            executor.shutdown()

        while not executor.isTerminated():
            continue

        if (executor.isTerminated()):

            end = time.time()
            print(f"Time spent generating walks was: {end-start}")
            start = time.time()

            
            #Generation of embeddings:
            self.gen_embeddings()
            
            end = time.time()
            print(f"Time spent generating embeddings was: {end-start}")

    def load_corpus(self):
        file_corpus = open(self.corpus_file_path, 'r')
        corpus = [line.rstrip('\n').split(' ') for line in file_corpus.readlines()]
        return corpus

    def gen_embeddings(self):
        
        print("Generating embeddings...")
        corpus = LineSentence(self.corpus_file_path)
     
        workers = os.cpu_count()
        model = Word2Vec(corpus, vector_size=self.embedding_size, window=self.window, min_count=self.min_count, sg=1, hs=1, workers=workers)

        word_vectors = model.wv
        word_vectors.save_word2vec_format(f"{self.data_root}/embeddings_readable.txt")
        word_vectors.save(self.embeddings_file_path)


    def train(self):

        graph = self.gen_graph()
        return graph
        #self.generate_corpus_and_embeddings(graph)




##########################################
        # EVALUATION FUNCTIONS
##########################################

    class Pair():
        
        def __init__(self, node1, node2, score = "0"):
            self.node1 = node1
            self.node2 = node2
            self.score = score

        def __repr__(self):
            return '\t'.join((self.node1, self.node2, str(self.score)))

        def __eq__(self, other):
            on_node1 = self.node1 == other.node1
            on_node2 = self.node2 == other.node2

            return on_node1 and on_node2

        def __key(self):
            return (self.node1, self.node2)

        def __hash__(self):
            return hash(self.__key())



    def generate_predictions(self, relations):
        print("Generating predictions...")
        start = time.time()
        n_cores = os.cpu_count()
        embeddings = KeyedVectors.load(self.embeddings_file_path)
        vocab = embeddings.index_to_key
        dict_vocab = HashMap() #n_cores, 0.75, n_cores)
        
        for word in vocab:
            dict_vocab.put(word, ArrayList(list(embeddings.get_vector(word))))

        preds = ArrayList()

        executor =  Executors.newFixedThreadPool(n_cores)


        print("\tStarting parallel tasks...")
        with jpype.synchronized(preds):
            with jpype.synchronized(dict_vocab):
                for word1 in vocab:
                    worker = GenPred(word1, dict_vocab, relations ,preds)
                    executor.execute(worker)
                    
                executor.shutdown()

        while not executor.isTerminated():
            continue

        preds_concat = ArrayList()
        for p in preds:
            preds_concat.addAll(p)


        end = time.time()
        print(f"Predictions generated in {end-start} seconds")
        return preds


    def format_test_set(self):
        print("Formatting ground truth data set...")
        test_set = self.dataset.testing

        test_set = [[x[0][1:-1], x[1][1:-1], "0"] for x in test_set]
        _test_set = ArrayList()
        
        for x in test_set:
            _test_set.add(ArrayList(x))
        #_test_set.add(ArrayList(["a", "b", "0"]))
        return _test_set
 

    def compute_metrics(self, k, relations):
        #Computes hits@k and AUC
        #Input
        #    * k: value at which the rank is computed
        #    * relations: list of relations to compute the metrics. (The metrics are computed relation-wise)
        

        preds = self.generate_predictions(relations) # list (node 1, rel, node 2, score)

        ground_truth = self.format_test_set() # list (node 1, rel, node 2, score)
        print("Predictions: ", len(preds))
        print("Ground truth: ", len(ground_truth))

        
        #### BOTTLENECK
        start = time.time()
        entities = ArrayList()
        entities_ = {pair[0] for pair in preds + ground_truth}.union({pair[1] for pair in preds + ground_truth})
        for node in entities_:
            entities.add(node)

        dict_subj_hits = HashMap()
        dict_subj_ranks = HashMap()
        end = time.time()
        print(f"Time in bottleneck is {end-start}")
        ############


        print("Started evaluation...")
        start = time.time()
        n_cores = os.cpu_count()

        executor =  Executors.newFixedThreadPool(n_cores)

        with jpype.synchronized(ground_truth):
            with jpype.synchronized(preds):
                with jpype.synchronized(entities):
                    with jpype.synchronized(dict_subj_hits): 
                        with jpype.synchronized(dict_subj_ranks):
                            for pair in ground_truth:
                                worker = WROEval(pair, k, relations, ground_truth, preds, entities, dict_subj_hits, dict_subj_ranks)
                                executor.execute(worker)
                                
                            executor.shutdown()

        while not executor.isTerminated():
            continue

        if (executor.isTerminated()):
            end = time.time()
            print(f"Evaluation finished in {end-start} seconds")
            # do smthng


        results = {}

        for rel in relations:

            hits = dict_subj_hits[rel].values()
            hits =  reduce(lambda x,y: x+y, hits)

            ranks = dict_subj_ranks[rel].values()
            ranks = list(map(lambda x: Counter(x), ranks))
            ranks = dict(reduce(lambda x,y: x+y, ranks))

            preds_rel = [(n1, curr_rel , n2) for (n1, curr_rel, n2) in preds if rel==curr_rel]
           
            rank_auc = self.compute_rank_roc(ranks,len(entities))
            results[rel] = {f"hits_{k}": hits/len(preds_rel), "rank_auc": rank_auc}
            
        return results


    def compute_rank_roc(self, ranks, n_entities):

        auc_x = list(ranks.keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        sum_rank = sum(ranks.values())
        for x in auc_x:
            tpr += ranks[x]
            auc_y.append(tpr / sum_rank)
        auc_x.append(n_entities)
        auc_y.append(1)
        auc = np.trapz(auc_y, auc_x) 
        return auc/n_entities

    def evaluate(self, relations):
        print(self.compute_metrics(10, relations))

