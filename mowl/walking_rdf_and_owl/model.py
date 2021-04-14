from mowl.model import Model

import os
import time
import numpy as np
import gzip
import subprocess

import jpype
import jpype.imports

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from java.lang import Runnable, Thread
from java.io import PipedInputStream, PipedOutputStream, File

from org.semanticweb.owlapi.model import IRI
from org.apache.jena.rdf.model import ModelFactory
from org.apache.jena.util import FileManager

from java.util import HashMap, ArrayList
from java.util.concurrent import ExecutorService  
from java.util.concurrent import Executors  

from org.mowl import WorkerThread


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
            
            if count >100:
                break
            count+=1
           
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
        model = Word2Vec(corpus, size=self.embedding_size, window=self.window, min_count=self.min_count, sg=1, hs=1, workers=workers)

        # word_vectors = model.wv
# 
        # word_vectors.save("data/embeddings.wordvectors")

        model.wv.save_word2vec_format(self.embeddings_file_path)

    def train(self):

        graph = self.gen_graph()
        self.generate_corpus_and_embeddings(graph)




##########################################
        # EVALUATION FUNCTIONS
##########################################

    def generate_predictions(self):
        
        embeddings = self.load_embeddings()
        

        words = embeddings.get_normed_vectors()

        distances = distances(words)

        # words = set(reduce(lambda a,b: a + b, corpus, []))

        # preds = []
        # for word1 in words:
        #     for word2 in words:
        #         if word1 != word2:
        #             similarity = model.similarity(word1, word2)
        # # list words
        #             preds.append((word1, word2, score))
        # for each word compute similarity against all other words.

        # save predictions
        return distances


    def compute_metrics(gt_file, pred_file, k):
        #Computes hits@k and AUC

        # Read test file (ground truth) and submission file (predictions).
        triplets_gt   = read_triplets_file(gt_file)
        triplets_pred = read_triplets_file(pred_file)

        entities = set([t.entity_1 for t in triplets_gt] + [t.entity_2 for t in triplets_gt] + [t.entity_1 for t in triplets_pred] + [t.entity_2 for t in triplets_pred])

        ent1_rels = {}

        for triplet_gt in triplets_gt:
            ent1, rel = triplet_gt.entity_1, triplet_gt.relation

            if (ent1, rel) in ent1_rels:
                continue

            #Extract triplets with fixed entity 1 and relation
            grouped_triplets_gt   = set(filter(lambda x: x.entity_1 == ent1 and x.relation == rel, triplets_gt))
            grouped_triplets_pred = set(filter(lambda x: x.entity_1 == ent1 and x.relation == rel, triplets_pred))

            all_triplets = ({Triplet(ent1, rel, ent2, score = 0) for ent2 in entities} - grouped_triplets_pred).union(grouped_triplets_pred)
            all_triplets = list(all_triplets)

            scores = [-x.score for x in all_triplets]   
            ranking = rankdata(scores, method='average')

            hits = 0
            ranks = {}
            
            for grouped_triplet in list(grouped_triplets_gt):
                idx = all_triplets.index(grouped_triplet)
                rank = ranking[idx]

                if rank <= k:
                    hits+=1
                if not rank in ranks:
                    ranks[rank] = 0
                ranks[rank] += 1

            ent1_rels[(ent1, rel)] = (hits, ranks)


        hits = map(lambda x: x[0], ent1_rels.values())
        hits =  reduce(lambda x,y: x+y, hits)

        ranks = map(lambda x: x[1], ent1_rels.values())
        ranks = list(map(lambda x: Counter(x), ranks))
        ranks = dict(reduce(lambda x,y: x+y, ranks))

        result = hits/len(triplets_pred)

        rank_auc = compute_rank_roc(ranks,len(entities))

        return result, rank_auc


    def compute_rank_roc(ranks, n_entities):

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