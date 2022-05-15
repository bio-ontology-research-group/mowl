from org.semanticweb.owlapi.apibinding import OWLManager
from uk.ac.manchester.cs.owl.owlapi import OWLSubClassOfAxiomImpl, OWLDisjointClassesAxiomImpl, OWLEquivalentClassesAxiomImpl
from java.util import HashSet
from functools import wraps


import logging
logging.basicConfig(level=logging.INFO)


            
def count_added_axioms(func):
    @wraps(func)
    def wrapper(self, ontology):
        initial_number = ontology.getAxiomCount()
        func(self, ontology)
        final_number = ontology.getAxiomCount()
        logging.info(f"Initial axioms: {initial_number}. Final axioms: {final_number}. Added: {final_number - initial_number}.")

    return wrapper
    


class MOWLReasoner():
    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.ont_manager = OWLManager.createOWLOntologyManager()

    @count_added_axioms
    def infer_subclass_axioms(self, ontology):
        ont_classes = ontology.getClassesInSignature(True)
        for ont_class in ont_classes:
            super_classes = self.reasoner.getSuperClasses(ont_class, False).getFlattened()
            new_axioms = set(map(lambda x: OWLSubClassOfAxiomImpl(ont_class, x, []), super_classes))

            axiom_set = HashSet()
            axiom_set.addAll(new_axioms)
            self.ont_manager.addAxioms(ontology, axiom_set)

    @count_added_axioms
    def infer_equiv_class_axioms(self, ontology):
        ont_classes = ontology.getClassesInSignature(True)
        for ont_class in ont_classes:
            equiv_classes = self.reasoner.getEquivalentClasses(ont_class).getEntities()
            equiv_classes.add(ont_class)

            new_axiom = OWLEquivalentClassesAxiomImpl(equiv_classes, [])
#            new_axioms = list(map(lambda x: OWLEquivalentClassesAxiomImpl(ont_class, x, []), equiv_classes))

 #           axiom_set = HashSet()
 #           axiom_set.add(ont_class)
 #           axiom_set.addAll(new_axioms)

            self.ont_manager.addAxiom(ontology, new_axiom)
            
    @count_added_axioms
    def infer_disjoint_class_axioms(self, ontology):
        ont_classes = ontology.getClassesInSignature(True)
        for ont_class in ont_classes:
            disjoint_classes = self.reasoner.getDisjointClasses(ont_class).getFlattened()
            new_axioms = list(map(lambda x: OWLDisjointClassesAxiomImpl(ont_class, x, []), disjoint_classes))

            axiom_set = HashSet()
            axiom_set.addAll(new_axioms)
            self.ont_manager.addAxioms(ontology, axiom_set)

            


