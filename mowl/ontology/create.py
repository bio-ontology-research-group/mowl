import logging
import os
from jpype import *
import jpype.imports


from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI

def create_from_triples(
        triples_file,
        out_file,
        relation_name = None,
        bidirectional = False,
        head_prefix = "",
        tail_prefix = ""
):

    """Method to create an ontology from a .tsv file with triples.    

    :param triples_file: Path for the file containing the triples. This file must be a `.tsv` file and each row must be of the form (head, relation, tail). It is also supported `.tsv` files with rows of the form (head, tail); in that case the field `relation_name` must be specified.
    :type triples_file: str
    :param relation_name: Name for relation in case the `.tsv` input file has only two columns.
    :type relation_name: str
    :param bidirectional: If `True`, the triples will be considered undirected.
    :type bidirectional: bool
    :param out_file: Path for the output ontology. If `None` and an existing ontology is input, the existing ontology will be overwritten.
    :type out_file: str
    :param head_prefix: Prefix to be assigned to the head of each triple. Default is `""`
    :type head_prefix: str
    :param tail_prefix: Prefix to be assigned to the tail of each triple. Default is `""`
    :type tail_prefix: str
    """

    manager = OWLManager.createOWLOntologyManager()
    factory = manager.getOWLDataFactory()

    ont = manager.createOntology()

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

            head = factory.getOWLClass(IRI.create(f"{head_prefix}{head}"))
            rel = factory.getOWLObjectProperty(IRI.create(f"{rel}"))
            tail = factory.getOWLClass(IRI.create(f"{tail_prefix}{tail}"))

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

    

    
                                          
