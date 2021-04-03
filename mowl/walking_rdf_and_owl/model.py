from mowl.model import Model

import os
import jpype
import jpype.imports
from java.lang import Runnable, Thread
from java.io import PipedInputStream, PipedOutputStream, File
from org.semanticweb.owlapi.model import IRI
from org.apache.jena.rdf.model import ModelFactory
from org.apache.jena.util import FileManager

jars_dir = "../gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'


if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jars,
        convertStrings=False)

class WalkRdfOwl(Model):
    def __init__(self, dataset):
        super().__init__(dataset)    


    def gen_graph(self, output, mapping, undirected, format= "RDF/XML"):

        tmp_data_file = File.createTempFile(os.getcwd() + '/data/temp_file', '.tmp')

        output = "testing____.txt"
        withMap = True

 #       tmp_data_file = java.io.File.createTempFile(os.getcwd() + '/data/temp_file', '.tmp')

#        ont_manager = OWLManager.createOWLOntologyManager()

        # ontology_set_files = os.listdir(ontology_directory)
        # ont_hash_set = java.util.LinkedHashSet()
        # for file in ontology_set_files:
        #     ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(java.io.File(os.getcwd()+'/'+ontology_directory+file)))
        # ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(java.io.File(os.getcwd()+'/'+input)))

        # ontology = ont_manager.createOntology(IRI.create("http://aber-owl.net/rdfwalker/t.owl"), ont_hash_set)

        # factory =  ont_manager.getOWLDataFactory()

        # progressMonitor = ConsoleProgressMonitor()
        # config = SimpleConfiguration(progressMonitor)
        
        # reasoner_factory = ElkReasonerFactory()
        # reasoner = reasoner_factory.createReasoner(ontology,config)

#
#        inferred_axioms = InferredClassAssertionAxiomGenerator().createAxioms(factory, reasoner)

        self.dataset.infer_axioms()

        self.dataset.ont_manager.saveOntology(self.dataset.ontology, IRI.create(tmp_data_file.toURI()))



       ## ooo = PipedOutputStream()
       # iii = PipedInputStream()
       # iii.connect(ooo)
       # self.dataset.ontology.saveOntology(ooo)

       # iii.read()
 
        # counter = 0
        # for axiom in inferred_axioms:
        #     ont_manager.addAxiom(ontology, axiom)
        #     counter += 1
            
        # ont_manager.saveOntology(ontology, IRI.create(tmp_data_file.toURI()))


        #print(str(counter) + " axioms inferred.")

    #     print("After infering axioms")

    #     @jpype.JImplements(Runnable)
    #     class OutputStrm():
    #    #     jpype.attachThreadToJVM()
    #         def __init__(self, ontology):
    #             self.ontology = ontology
    #             self.outputStream = PipedOutputStream()
                
    #         @jpype.JOverride
    #         def run(self):
    #             threadId = Thread.currentThread().getId()
    #             print(f"From thread output id is {threadId}")
    #             ontology.saveOntology(self.outputStream)
    #             print("Output Thread after business")

    #     @jpype.JImplements(Runnable)
    #     class InputStrm():
    #    #     jpype.attachThreadToJVM()
    #         def __init__(self):
    #             self.inputStream = PipedInputStream()
    #           #  self.outputStream = outputStream
    #            # self.inputStream.connect(self.outputStream)
                
    #         @jpype.JOverride
    #         def run(self):
    #             threadId = Thread.currentThread().getId()
    #             print(f"From thread input id is {threadId}")
    #             self.inputStream.read()
    #             print("Input Thread after business")

    #     print("Before asking for attached thread")
    #     if not jpype.isThreadAttachedToJVM():
    #         jpype.attachThreadToJVM()
    #     print("After asking for attached thread")
        


    #     outputStrm = OutputStrm(self.dataset.ontology)
    #     print("Create outputStrm")
    #     inputStrm = InputStrm()
    #     print("Create inputStrm")

    #     inputStrm.inputStream.connect(outputStrm.outputStream)
    #     print("Connect inputStrm")

    #     thread1 = Thread(outputStrm)
    #     thread1.start()
    #     thread1.join()

    #     print("Started thread1")

    #     thread2 = Thread(inputStrm)
    #     thread2.start()
    #     print("Started thread2")


    #     print("Joined thread 1")

    #     thread2.join()

    #     print("Joined thread 2")
        filename = tmp_data_file.toURI()
        model = ModelFactory.createDefaultModel()
        infile = FileManager.get().open(filename.toString())

        model.read(infile, None, format)

        # outputStrm.outputStream.close()
        # inputStrm.inputStream.close()

        counter = 1
        dict = {} # maps IRIs to ints; for input to deepwalk

        out_file = open(output, 'w')
        for stmt in model.listStatements():
            pred = stmt.getPredicate()
            subj = stmt.getSubject()
            obj = stmt.getObject()

            
            if withMap:

                if (subj.isURIResource() and obj.isURIResource()):
                
                    if not pred in dict:
                        dict[pred] = counter
                        counter += 1
                    
                    if not subj in dict:
                        dict[subj] = counter
                        counter += 1
                
                    if not obj in dict:
                        dict[obj] = counter
                        counter += 1
                    
                    predid = dict[pred]
                    subjid = dict[subj]
                    objid = dict[obj]
                    
                    # generate three nodes and directed edges
                    out_file.write(str(subjid)+"\t"+str(objid)+"\t"+str(predid)+"\n")
                    
                    # add reverse edges for undirected graph; need to double the walk length!
                    if (undirected):
                        out_file.write(str(objid)+"\t"+str(subjid)+"\t"+str(predid)+"\n")

            else:
                if (subj.isURIResource() and obj.isURIResource()):
                    
                    # generate three nodes and directed edges
                    out_file.write(subj+"\t"+obj+"\t"+pred+"\n")
                    
                    # add reverse edges for undirected graph; need to double the walk length!
                    if (undirected):
                        out_file.write(obj+"\t"+subj+"\t"+pred+"\n")                        

        map_file = open(mapping, 'w')

        if (withMap):
            for key, val in dict.items():
                map_file.write(str(key)+'\t'+str(val)+'\n')