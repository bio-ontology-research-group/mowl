from mowl.model import Model

import os
import time
import numpy as np
import gzip
import subprocess
from functools import reduce
import operator
from scipy.stats import rankdata
from collections import Counter

import jpype
import jpype.imports

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

from java.lang import Runnable, Thread
from java.io import PipedInputStream, PipedOutputStream, File

from org.semanticweb.owlapi.model import IRI
from org.apache.jena.rdf.model import ModelFactory
from org.apache.jena.util import FileManager

from java.util import HashMap, ArrayList
from java.util.concurrent import ExecutorService  
from java.util.concurrent import Executors  

from org.mowl import WorkerThread, WROEval


jars_dir = "../gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'


if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jars,
        convertStrings=False)

class WalkRdfOwl(Model):
    def __init__(self, 
                    dataset, 
                    corpus_file_path, 
                    embeddings_file_path, 
                    number_walks, 
                    length_walk, 
                    undirected=True, 
                    embedding_size=100, 
                    window=3, 
                    min_count = 5):

        super().__init__(dataset)    
        self.corpus_file_path = corpus_file_path
        self.embeddings_file_path = embeddings_file_path
        self.number_walks = number_walks
        self.length_walk = length_walk
        self.undirected = undirected

        # Skip-gram params
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count

    def gen_graph(self, format= "RDF/XML"):

        tmp_data_file = File.createTempFile(os.getcwd() + '/data/temp_file', '.tmp')
    

        self.dataset.infer_axioms()
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
                pred = str(stmt.getPredicate())
                subj = str(stmt.getSubject())
                obj =  str(stmt.getObject())

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
        word_vectors.save_word2vec_format("data/embeddings_readable.txt")
        word_vectors.save(self.embeddings_file_path)


    def train(self):

        graph = self.gen_graph()
        self.generate_corpus_and_embeddings(graph)




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

    def generate_predictions(self):
        print("Generating predictions...")
        num = 1000
        embeddings = KeyedVectors.load(self.embeddings_file_path)
        vocab = embeddings.index_to_key
        preds = ArrayList()
        for i in range(len(vocab)):
            word1 = vocab[i]
            if 'http://4932.' in word1 and num >0:
                num -=1
                for j in range(len(vocab)):
                    word2 = vocab[j]
                    if word1 != word2 and 'http://4932.' in word2:
                        similarity = embeddings.similarity(word1, word2)
                        preds.add(ArrayList([word1, word2, str(similarity)]))
            elif num <= 0:
                break
        return preds


    def format_test_set(self):
        print("Formatting ground truth data set...")
        test_set = self.dataset.testing

        test_set = np.delete(test_set, 1, 1) # remove column with index 1. This column corresponds to the relation
        test_set = [[x[0][1:-1], x[1][1:-1], "0"] for x in test_set]
        _test_set = ArrayList()
        
        for x in test_set:
            _test_set.add(ArrayList(x))
        #_test_set.add(ArrayList(["a", "b", "0"]))
        return _test_set
 

    def compute_metrics(self, k):
        #Computes hits@k and AUC

        preds = self.generate_predictions() # list (node 1, node 2, score)

        ground_truth = self.format_test_set() # list (node 1, node 2, score)
        print("GROUND TRUTH: ", len(preds))
        print("GROUND TRUTH: ", len(ground_truth))

        
        entities = ArrayList()
        entities_ = set([pair[0] for pair in preds] + [pair[1] for pair in preds] + [pair[0] for pair in ground_truth] + [pair[1] for pair in ground_truth])
        for node in entities_:
            entities.add(node)

        dict_subj_hits = HashMap()
        dict_subj_ranks = HashMap()

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
                                worker = WROEval(pair, k, ground_truth, preds, entities, dict_subj_hits, dict_subj_ranks)
                                executor.execute(worker)
                                
                            executor.shutdown()

        while not executor.isTerminated():
            continue

        if (executor.isTerminated()):
            end = time.time()
            print(f"Evaluation finished in {end-start} seconds")
            # do smthng

#        for triplet_gt in triplets_gt:
        # for pair in ground_truth:
        #     #ent1, rel = triplet_gt.entity_1, triplet_gt.relation
        #     node1 = pair.node1
        #     if node1 in entities_1:
        #         continue

        #     #Extract triplets with fixed entity 1
        #     grouped_pairs_gt   = {x for x in ground_truth if x.node1 == node1} #set(filter(lambda x: x.node1 == node1 , ground_truth))
        #     grouped_pairs_pred = {x for x in preds if x.node1 == node1} #set(filter(lambda x: x.node1 == node1 , preds))
        

        #    # print(f"Length preds: {len(grouped_pairs_pred)}")
        #     all_pairs = ({self.Pair(node1, ent2, 0) for ent2 in entities} - grouped_pairs_pred).union(grouped_pairs_pred)
        #     all_pairs = list(all_pairs)

        #     scores = [-x.score for x in all_pairs]   
        #     ranking = rankdata(scores, method='average')

        #     hits = 0
        #     ranks = {}
            
        #     for grouped_pair in list(grouped_pairs_gt):
        #         idx = all_pairs.index(grouped_pair)
        #         rank = ranking[idx]
        #         if(scores[idx] > 0):
        #             print(f"Rank is {rank}. Score is {scores[idx]}")
        #         if rank <= k:
        #             hits+=1
        #         if not rank in ranks:
        #             ranks[rank] = 0
        #         ranks[rank] += 1

        #     entities_1[node1] = (hits, ranks)


        hits = dict_subj_hits.values()
        hits =  reduce(lambda x,y: x+y, hits)

        ranks = dict_subj_ranks.values()
        ranks = list(map(lambda x: Counter(x), ranks))
        ranks = dict(reduce(lambda x,y: x+y, ranks))

        result = hits/len(preds)

        rank_auc = self.compute_rank_roc(ranks,len(entities))

        return result, rank_auc


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

    def evaluate(self):
        print(self.compute_metrics(3))