from org.semanticweb.owlapi.manchestersyntax.renderer import ManchesterOWLSyntaxOWLObjectRendererImpl
from org.semanticweb.owlapi.model import OWLLiteral
from org.semanticweb.owlapi.search import EntitySearcher


from jpype.types import *

from org.mowl import MOWLShortFormProvider
import logging



def extract_and_save_axiom_corpus(ontology, out_file):
        
    logging.info("Generating axioms corpus")
    renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
    shortFormProvider = MOWLShortFormProvider()
    renderer.setShortFormProvider(shortFormProvider)
    with open(out_file, 'w') as f:
        for owl_class in ontology.getClassesInSignature():
            axioms = ontology.getAxioms(owl_class)
            for axiom in axioms:
                rax = renderer.render(axiom)
                rax = rax.replaceAll(JString("[\\r\\n|\\r|\\n()|<|>]"), JString(""))
                f.write(f'{rax}\n')

def extract_axiom_corpus(ontology):

    logging.info("Generating axioms corpus")
    renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
    shortFormProvider = MOWLShortFormProvider()
    renderer.setShortFormProvider(shortFormProvider)

    corpus = []
    
    for owl_class in ontology.getClassesInSignature():
        axioms = ontology.getAxioms(owl_class)
        for axiom in axioms:
            rax = renderer.render(axiom)
            rax = rax.replaceAll(JString("[\\r\\n|\\r|\\n()|<|>]"), JString(""))
            corpus.append(rax)
    return corpus



def extract_annotation_corpus(ontology, out_file, mode = "append"):

    if mode == "append":
        mode = "a"
    else:
        mode = "w"
    with open(out_file, mode) as f:
        for owl_class in ontology.getClassesInSignature():
            cls = str(owl_class)
              
            annotations = EntitySearcher.getAnnotations(owl_class, ontology)
            for annotation in annotations:
                if isinstance(annotation.getValue(), OWLLiteral):
                    property = str(annotation.getProperty()).replace("\n", " ")
                    # could filter on property
                    value = str(annotation.getValue().getLiteral()).replace("\n", " ")
                    f.write(f'{cls} {property} {value}\n')
