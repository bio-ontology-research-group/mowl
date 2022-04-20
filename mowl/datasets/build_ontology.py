import logging
import os
from jpype import *
import jpype.imports


from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI

def insert_annotations(ontology_file, annotations, out_file = None, verbose=False):
    """
    Method to build dataset given an ontology file and the annotations that must be attached to the ontology. Annotation files must be in .tsv format, with no header. Per each row, the first element is the annotated entity and the rest of the elements are the annotation entities (which are the entities in the ontology).

    :param ontology_file: Ontology file in .owl format
    :type ontology_file: str
    :param annotations: Annotations to be included in the ontology. There can be more than one annotation file.
    :type annotations: List of (str, str) corresponding to (annotation file path, relation name)
    :param out_file: Path for the new ontology.
    :type out_file: str
    :param verbose: If true, information is shown."
    :type verbose: bool
    """

            
    if verbose:
        logging.basicConfig(level = logging.INFO)

    if out_file is None:
        out_file = ontology_file
        
        
    manager = OWLManager.createOWLOntologyManager()
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(ontology_file))
    factory = manager.getOWLDataFactory()

    for annots_file, relation_name in annotations:
        relation = factory.getOWLObjectProperty(IRI.create(f"http://{relation_name}"))

        with open(annots_file) as f:
            for line in f:
                items = line.strip().split("\t")
                annotated_entity = items[0]
                annotated_entity = factory.getOWLClass(IRI.create(f"http://{annotated_entity}"))

                for ont_id in items[1:]:
#                    ont_id = ont_id.replace(":", "_")
                    ont_class = factory.getOWLClass(IRI.create(f"{ont_id}"))
                    objSomeValsAxiom = factory.getOWLObjectSomeValuesFrom(relation, ont_class)
                    axiom = factory.getOWLSubClassOfAxiom(annotated_entity, objSomeValsAxiom)
                    manager.addAxiom(ont, axiom)

    
    manager.saveOntology(ont, IRI.create("file:" + os.path.abspath(out_file)))



def add_triples(triples_file, relation_name = None, out_file = None , bidirectional = False, current_ontology_file = None):

    """Method to create an ontology from a .tsv file with triplets.    

    :param triples_file: Path for the file containing the triples. This file must be a `.tsv` file and each row must be of the form (head, relation, tail). It is also supported `.tsv` files with rows of the form (head, tail); in that case the field `relation_name` must be specified.
    :type triples_file: str
    :param relation_name: Name for relation in case the `.tsv` input file has only two columns.
    :type relation_name: str
    :param current_ontology_file: Use this parameter to add the triples to an existing ontology. Otherwise, a new ontology will be created.
    :type current_ontology_file: str
    :param bidirectional: If `True`, the triples will be considered undirected.
    :type bidirectional: bool
    :param out_file: Path for the output ontology. If `None` and an existing ontology is input, the existing ontology will be overwritten.
    :type out_file: str
    """

    if out_file is None and current_ontology_file is None:
        raise ValueError("Neither output file nor current ontology file were not specified ")

    
    manager = OWLManager.createOWLOntologyManager()
    factory = manager.getOWLDataFactory()

    root_dir = triples_file.split("/")[:-1]
    root_dir = "/".join(root_dir) + "/"

    if current_ontology_file is None:
        ont = manager.createOntology()
    else:
        if out_file is None:
            out_file = current_ontology_file
        ont = manager.loadOntologyFromOntologyDocument(java.io.File(current_ontology_file))

    

    with open(triples_file, "r") as f:
        for line in f:
            line = tuple(line.strip().split("\t"))

            if len(line) < 2 or len(line) > 3:
                raise ValueError(f"Expected number of elements in triple to be 2 or 3. Got {len(line)}")
            if len(line) == 2 and relation_name is None:
                raise ValueError("Found 2 elements in triple but the relation_name field is None")

            if len(line) == 2:
                head, tail = line
                rel = relation_name
            if len(line) == 3:
                head, rel, tail = line

            head = factory.getOWLClass(IRI.create(f"http://{head}"))
            rel = factory.getOWLObjectProperty(IRI.create(f"http://{rel}"))
            tail = factory.getOWLClass(IRI.create(f"http://{tail}"))

            axiom = factory.getOWLSubClassOfAxiom(
                head, factory.getOWLObjectSomeValuesFrom(
                    rel, tail))
            manager.addAxiom(ont, axiom)

            if bidirectional:
                axiom = factory.getOWLSubClassOfAxiom(
                    tail, factory.getOWLObjectSomeValuesFrom(
                        rel, head))
                manager.addAxiom(ont, axiom)
                

    manager.saveOntology(ont, IRI.create("file:" + os.path.abspath(out_file)))

    

    
