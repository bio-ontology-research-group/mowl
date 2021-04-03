from mowl.model import Model

import os
import jpype
import jpype.imports

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
    def __init__(self, dataset, out_file_path, number_walks, length_walk, undirected=True):
        super().__init__(dataset)    
        self.out_file_path = out_file_path
        self.number_walks = number_walks
        self.length_walk = length_walk
        self.undirected = undirected


    def gen_graph(self, format= "RDF/XML"):

        tmp_data_file = File.createTempFile(os.getcwd() + '/data/temp_file', '.tmp')
    

        self.dataset.infer_axioms()
        self.dataset.ont_manager.saveOntology(self.dataset.ontology, IRI.create(tmp_data_file.toURI()))


        filename = tmp_data_file.toURI()
        model = ModelFactory.createDefaultModel()
        infile = FileManager.get().open(filename.toString())

        model.read(infile, None, format)        

        edge_list = HashMap()

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


    def generate_corpus(self, graph):
        
        print("Starting walking...")

        n_cores = os.cpu_count()

        executor =  Executors.newFixedThreadPool(n_cores)

        #Just to clear the file before writing again on it.
        f = open(self.out_file_path, 'w')
        f.close()

        sources = ArrayList(graph.keySet())
        
        out_file_j = jpype.JObject(self.out_file_path, jpype.JClass("java.lang.String"))

        with jpype.synchronized(graph):
            for i in range(len(sources)):
                worker = WorkerThread(out_file_j, graph, self.number_walks, self.length_walk, sources[i])
                executor.execute(worker)
                
            executor.shutdown()


    def train(self):

        graph = self.gen_graph()
        self.generate_corpus(graph)