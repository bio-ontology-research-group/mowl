import mowl
mowl.init_jvm("20g")
from mowl.reasoning import MOWLReasoner
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat
from org.semanticweb.owlapi.model import AxiomType as AT
from org.semanticweb.owlapi.model import ClassExpressionType as CT
from org.semanticweb.owlapi.model.parameters import Imports

from java.util import HashSet

import os
import click as ck

@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):

    if not input_ontology.endswith(".owl"):
        raise ValueError("The input ontology must be in OWL format")

    ds = PathDataset(input_ontology)

    reasoner_factory = ElkReasonerFactory()
    reasoner = reasoner_factory.createReasoner(ds.ontology)
    mowl_reasoner = MOWLReasoner(reasoner)

    classes = list(ds.ontology.getClassesInSignature())
    original_axioms = set()

    for axiom in ds.ontology.getTBoxAxioms(Imports.fromBoolean(True)):
        if axiom.getAxiomType() == AT.SUBCLASS_OF:
            if axiom.getSubClass().getClassExpressionType() == CT.OWL_CLASS and axiom.getSuperClass().getClassExpressionType() == CT.OWL_CLASS:
                original_axioms.add(axiom)

    
    subclass_axioms = mowl_reasoner.infer_subclass_axioms(classes)
    print("Number of inferred subclass axioms: ", len(subclass_axioms))

    subclass_axioms_with_no_top = set()
    for axiom in subclass_axioms:
        super_class = axiom.getSuperClass()
        if super_class.isOWLThing():
            continue
        subclass_axioms_with_no_top.add(axiom)
    print("Number of inferred subclass axioms without owl:Thing: ", len(subclass_axioms_with_no_top))

    only_deductive_closure_axioms = subclass_axioms_with_no_top - original_axioms

    print("Number of axioms ONLY in the deductive closure: ", len(only_deductive_closure_axioms))
    
    hash_set = HashSet()
    hash_set.addAll(only_deductive_closure_axioms)
    
    output_file = input_ontology.replace(".owl", "_deductive_closure.owl")

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager

    ontology = manager.createOntology(hash_set)
    manager.saveOntology(ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(output_file)))


if __name__ == "__main__":
    main()
