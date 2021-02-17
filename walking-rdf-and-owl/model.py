from py4j.java_gateway import JavaGateway
from py4j.java_gateway import java_import

import os
import click as ck

@ck.command()
@ck.option(
    '--input', '-i', default='data/bio-knowledge-graph.nt',
    help='input RDF file')
@ck.option(
    '--output', '-o', default='edgelist.txt',
    help='output file to use as input in DeepWalk algorithm')
@ck.option(
    '--mapping', '-m', default='mappingFile.txt',
    help='Output mapping file. Contains numerical ids for all entities')
@ck.option(
    '--undirected', '-u', default=False,
    help='Build undirected graph (default: false)')
@ck.option(
    '--classify', '-c', default=True,
    help= 'Wse an OWL reasoner to classify the RDF dataset (must be in RDF/XML) before graph generation (default: false)')
@ck.option( 
    '--format', '-f', default= 'RDF/XML',
    help= 'RDF format; values are "RDF/XML", "N-TRIPLE", "TURTLE" and "N3" (default: RDF/XML)')
@ck.option(
    '--ontology-directory', '-ont-dir', default='data/ontology-dir/',
    help='Directory with ontologies to use for reasoning')


def main(input, output, mapping, undirected, classify, format, ontology_directory):

    gateway = JavaGateway()
    java_import(gateway.jvm, 'org.semanticweb')

    tmp_data_file = 'data/temp_file.tmp'

    if classify:
  
        ont_manager = gateway.jvm.org.semanticweb.owlapi.apibinding.OWLManager.createOWLOntologyManager()

        ontology_set_files = os.listdir(ontology_directory)
        ont_hash_set = gateway.jvm.java.util.LinkedHashSet()
        for file in ontology_set_files:
            ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(gateway.jvm.java.io.File(ontology_directory+file)))
        ont_hash_set.add(ont_manager.loadOntologyFromOntologyDocument(gateway.jvm.java.io.File(input)))

        ontology = ont_manager.createOntology(gateway.jvm.org.semanticweb.owlapi.model.IRI.create("http://aber-owl.net/rdfwalker/t.owl"), ont_hash_set)

        factory =  ont_manager.getOWLDataFactory()

        progressMonitor = gateway.jvm.org.semanticweb.owlapi.reasoner.ConsoleProgressMonitor()
        config = gateway.jvm.org.semanticweb.owlapi.reasoner.SimpleConfiguration(progressMonitor)
        
        reasoner_factory = gateway.jvm.org.semanticweb.elk.owlapi.ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(ontology,config)

        inferred_axioms = gateway.jvm.orgorg.semanticweb.owlapi.util.InferredClassAssertionAxiomGenerator.createAxioms(factory, reasoner)
        counter = 0
        for axiom in inferred_axioms:
            ont_manager.addAxiom(ontology, axiom)
            counter += 1
        
        ont_manager.saveOntology(ontology, gateway.jvm.org.semanticweb.owlapi.model.IRI.create(tmp_data_file))
    

    if classify:
        filename = tmp_data_file
    else:
        filename = input


    model = gateway.jvm.org.apache.jena.rdf.model.ModelFactory().createDefaultModel()
    infile = gateway.jvm.org.apache.jena.util.FileManager.get().open( filename.toString())

    model.read(input, None, format)

    counter = 1
    dict = {} # maps IRIs to ints; for input to deepwalk

    out_file = open(output, 'w')
    for stmt in model.listStatements():
        pred = stmt.getPredicate()
        subj = stmt.getSubject()
        obj = stmt.getObject()

        if (subj.isURIResource() and obj.isURIResource()):
        
            if (dict[pred] == None):
                dict[pred] = counter
                counter += 1
            
            if (dict[subj] == None):
                dict[subj] = counter
                counter += 1
        
            if (dict[obj] == None):
                dict[obj] = counter
                counter += 1
            
            predid = dict[pred]
            subjid = dict[subj]
            objid = dict[obj]
            
            # generate three nodes and directed edges
            output.write(str(subjid)+"\t"+str(objid)+"\t"+predid+"\n")
            
            # add reverse edges for undirected graph; need to double the walk length!
            if (undirected):
                output.write(str(objid)+"\t"+str(subjid)+"\t"+predid+"\n")
            

    map_file = open(mapping, 'w')

    for key, val in dict.items():
        map_file.write(str(key)+'\t'+str(val)+'\n')

if __name__ == '__main__':
    main()