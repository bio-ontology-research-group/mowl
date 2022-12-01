import logging
import os
from jpype import java

from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI


def insert_annotations(ontology_file, annotations, out_file=None, verbose=False):
    """
    Method to build dataset given an ontology file and the annotations to be inserted to the
    ontology. Annotation files must be in .tsv format, with no header. Per each row, the first
    element is the annotated entity and the rest of the elements are the annotating entities
    (which are the entities in the ontology).

    :param ontology_file: Ontology file in .owl format
    :type ontology_file: str
    :param annotations: Annotations to be included in the ontology. There can be more than one
        annotation file.
    :type annotations: List of (str, str, bool) corresponding to (annotation file path, relation
        name, directed or undirected graph)
    :param out_file: Path for the new ontology. Defaults to ``ontology_file``
    :type out_file: str, optional
    :param verbose: If true, information is shown."
    :type verbose: bool
    """

    if not isinstance(ontology_file, str):
        raise TypeError("Parameter ontology_file must be of type str")
    if not isinstance(annotations, list):
        raise TypeError("Parameter annotations must be of type list")
    if out_file is not None and not isinstance(out_file, str):
        raise TypeError("Optional parameter out_file must be of type str")
    if not isinstance(verbose, bool):
        raise TypeError("Optional parameter verbose must be of type bool")

    if verbose:
        logging.basicConfig(level=logging.INFO)

    if out_file is None:
        out_file = ontology_file

    manager = OWLManager.createOWLOntologyManager()
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(ontology_file))

    factory = manager.getOWLDataFactory()

    for annots_file, relation_name, directed in annotations:
        relation = factory.getOWLObjectProperty(IRI.create(f"{relation_name}"))

        with open(annots_file) as f:
            for line in f:
                items = line.strip().split("\t")
                annotating_entity = items[0]
                annotating_entity = factory.getOWLClass(IRI.create(f"{annotating_entity}"))

                for ont_id in items[1:]:
                    ont_class = factory.getOWLClass(IRI.create(f"{ont_id}"))
                    objSomeValsAxiom = factory.getOWLObjectSomeValuesFrom(relation, ont_class)
                    axiom = factory.getOWLSubClassOfAxiom(annotating_entity, objSomeValsAxiom)
                    manager.addAxiom(ont, axiom)
                    if not directed:
                        objSomeValsAxiom = factory.getOWLObjectSomeValuesFrom(relation,
                                                                              annotating_entity)
                        axiom = factory.getOWLSubClassOfAxiom(ont_class, objSomeValsAxiom)
                        manager.addAxiom(ont, axiom)

    manager.saveOntology(ont, IRI.create("file:" + os.path.abspath(out_file)))
