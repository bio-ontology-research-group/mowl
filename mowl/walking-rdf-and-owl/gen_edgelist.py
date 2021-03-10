from pathlib import Path
from jpype import *
import jpype.imports
import os
import click as ck


jars_dir = "../gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir+name for name in os.listdir(jars_dir)])}'
startJVM(getDefaultJVMPath(), "-ea",  "-Djava.class.path=" + jars,  convertStrings=False)


from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.reasoner import ConsoleProgressMonitor
from org.semanticweb.owlapi.reasoner import SimpleConfiguration
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.util import InferredClassAssertionAxiomGenerator
from org.apache.jena.rdf.model import ModelFactory
from org.apache.jena.util import FileManager

@ck.command()
@ck.option(
    '--input', '-i', default='test_data/bio-knowledge-graph.nt',
    help='input RDF file')
@ck.option(
    '--output', '-o', default='test_data/edgelist.txt',
    help='output file to use as input in DeepWalk algorithm')
@ck.option(
    '--mapping', '-m', default='test_data/mappingFile.txt',
    help='Output mapping file. Contains numerical ids for all entities')
@ck.option(
    '--undirected/--directed', '-u/-d', default=False,
    help='Build undirected graph (default: false)')
@ck.option(
    '--classify/--no-classify', '-c/-nc', default=True,
    help= 'Wse an OWL reasoner to classify the RDF dataset (must be in RDF/XML) before graph generation (default: false)')
@ck.option( 
    '--format', '-f', default= 'RDF/XML',
    help= 'RDF format; values are "RDF/XML", "N-TRIPLE", "TURTLE" and "N3" (default: RDF/XML)')
@ck.option(
    '--ontology-directory', '-ont-dir', default='test_data/ontology-dir/',
    help='Directory with ontologies to use for reasoning')


def main(input, output, mapping, undirected, classify, format, ontology_directory):


    tmp_data_file = java.io.File.createTempFile(os.getcwd() + '/data/temp_file', '.tmp')

    ont_manager = OWLManager.createOWLOntologyManager()

    ontology_set_files = os.listdir(ontology_directory)
    ont_hash_set = java.util.LinkedHashSet()
    for file in ontology_set_files:
        ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(java.io.File(os.getcwd()+'/'+ontology_directory+file)))
    ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(java.io.File(os.getcwd()+'/'+input)))

    ontology = ont_manager.createOntology(IRI.create("http://aber-owl.net/rdfwalker/t.owl"), ont_hash_set)

    factory =  ont_manager.getOWLDataFactory()

    progressMonitor = ConsoleProgressMonitor()
    config = SimpleConfiguration(progressMonitor)
    
    reasoner_factory = ElkReasonerFactory()
    reasoner = reasoner_factory.createReasoner(ontology,config)

    inferred_axioms = InferredClassAssertionAxiomGenerator().createAxioms(factory, reasoner)
    counter = 0
    for axiom in inferred_axioms:
        ont_manager.addAxiom(ontology, axiom)
        counter += 1
        
    ont_manager.saveOntology(ontology, IRI.create(tmp_data_file.toURI()))


    print(str(counter) + " axioms inferred.")

    
    filename = tmp_data_file.toURI()
    model = ModelFactory.createDefaultModel()
    infile = FileManager.get().open(filename.toString())

    model.read(infile, None, format)

    counter = 1
    dict = {} # maps IRIs to ints; for input to deepwalk

    out_file = open(output, 'w')
    for stmt in model.listStatements():
        pred = stmt.getPredicate()
        subj = stmt.getSubject()
        obj = stmt.getObject()

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
            

    map_file = open(mapping, 'w')

    for key, val in dict.items():
        map_file.write(str(key)+'\t'+str(val)+'\n')

def URI(path):
    return "file:" + path

if __name__ == '__main__':

    main()
    shutdownJVM()
 