
from uk.ac.manchester.cs.owl.owlapi import OWLSubClassOfAxiomImpl, OWLDisjointClassesAxiomImpl, \
    OWLEquivalentClassesAxiomImpl
from org.semanticweb.owlapi.model import OWLClass
from org.semanticweb.owlapi.reasoner import OWLReasoner

from java.util import HashSet
from functools import wraps

from mowl.owlapi.adapter import OWLAPIAdapter

import logging
logging.basicConfig(level=logging.INFO)


def count_added_axioms(func):
    @wraps(func)
    def wrapper(self, owl_classes, *args, **kwargs):
        axioms = func(self, owl_classes, *args, **kwargs)
        logging.info(f"Number of inferred axioms: {len(axioms)}.")
        return axioms
    return wrapper


class MOWLReasoner():

    """This class encapsulates some of the funcionalities in the OWLAPI. It provies methods to \
        infer different types of axioms to extend a particular ontology.

    :param reasoner: Reasoner that will be used to infer the axioms.
    :type reasoner: This parameter has to implement the OWLAPI interface \
    :class:`org.semanticweb.owlapi.reasoner.OWLReasoner`
    """

    def __init__(self, reasoner):

        if not isinstance(reasoner, OWLReasoner):
            raise TypeError("Parameter reasoner must be an instance of \
org.semanticweb.owlapi.reasoner.OWLReasoner")

        self.reasoner = reasoner
        self.adapter = OWLAPIAdapter()
        self.ont_manager = self.adapter.owl_manager

    @count_added_axioms
    def infer_subclass_axioms(self, owl_classes, direct=False):
        """Infers and returns axioms of the type :math:`C \sqsubseteq D`

        :param owl_classes: List of OWLClass objects to be used to infer the axioms.
        :type owl_class: list[:class:`org.semanticweb.owlapi.model.OWLClass`]
        :param direct: If True, only direct superclasses will be inferred. Default is False.
        :type direct: bool, optional
        :rtype: list[:class:`org.semanticweb.owlapi.model.OWLSubClassOfAxiom`]
        """
        owl_classes = list(owl_classes)

        if not all(isinstance(owl_class, OWLClass) for owl_class in owl_classes):
            raise TypeError("All elements in parameter owl_classes must be of type \
org.semanticweb.owlapi.model.OWLClass")

        if not isinstance(direct, bool):
            raise TypeError("Optional parameter direct must be of type bool")

        axioms = []
        for owl_class in owl_classes:
            super_classes = self.reasoner.getSuperClasses(owl_class, direct).getFlattened()
            new_axioms = set(map(lambda x: OWLSubClassOfAxiomImpl(owl_class, x, []),
                                 super_classes))

            axioms += list(new_axioms)
        return axioms

    @count_added_axioms
    def infer_equivalent_class_axioms(self, owl_classes):
        """Infers and returns axioms of the form :math:`C \equiv D`

        :param owl_classes: List of OWLClass objects to be used to infer the axioms.
        :type owl_class: list[:class:`org.semanticweb.owlapi.model.OWLClass`]

        :rtype: list[:class:`org.semanticweb.owlapi.model.OWLEquivalentClassesAxiom`]
        """
        owl_classes = list(owl_classes)

        if not all(isinstance(owl_class, OWLClass) for owl_class in owl_classes):
            raise TypeError("All elements in parameter owl_classes must be of type \
org.semanticweb.owlapi.model.OWLClass")

        axioms = []
        for owl_class in owl_classes:
            equiv_classes = self.reasoner.getEquivalentClasses(owl_class).getEntities()
            equiv_classes.add(owl_class)

            new_axiom = OWLEquivalentClassesAxiomImpl(equiv_classes, [])
            axioms.append(new_axiom)
        return axioms

    @count_added_axioms
    def infer_disjoint_class_axioms(self, owl_classes):
        """Infers and adds axioms of the type :math:`C` *disjoint\_with*  :math:`D`

        :param owl_classes: List of OWLClass objects to be used to infer the axioms.
        :type owl_class: list[:class:`org.semanticweb.owlapi.model.OWLClass`]

        :rtype: list[:class:`org.semanticweb.owlapi.model.OWLDisjointClassesAxiom`]
        """
        owl_classes = list(owl_classes)

        if not all(isinstance(owl_class, OWLClass) for owl_class in owl_classes):
            raise TypeError("All elements in parameter owl_classes must be of type \
org.semanticweb.owlapi.model.OWLClass")

        axioms = []
        for owl_class in owl_classes:
            disjoint_classes = self.reasoner.getDisjointClasses(owl_class).getFlattened()
            disjoint_classes.add(owl_class)
            new_axiom = OWLDisjointClassesAxiomImpl(disjoint_classes, HashSet())
            axioms.append(new_axiom)
        return axioms
