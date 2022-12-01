"""This module implements shortcut methods to access some OWLAPI objects."""
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI

import mowl.error as err
from java.util import HashSet


class OWLAPIAdapter():
    """
    Adapter class adapting OWLAPI. Here you can find shortcuts to:
        - ``org.semanticweb.owlapi.apibinding.OWLManager``
        - ``org.semanticweb.owlapi.model.OWLDataFactory``
        - methods for creating OWLAPI objects

    """

    def __init__(self):

        self._owl_manager = None
        self._data_factory = None

    @property
    def owl_manager(self):
        """Creates a OWLManager from OWLAPI
        :rtype: org.semanticweb.owlapi.apibinding.OWLManager
        """
        if self._owl_manager is None:
            self._owl_manager = OWLManager.createOWLOntologyManager()

        return self._owl_manager

    @property
    def data_factory(self):
        """Creates an OWLDataFactory from OWLAPI. If OWLManager does not exist, it is created as \
            well.
        :rtype: org.semanticweb.owlapi.model.OWLDataFactory
        """

        if self._data_factory is None:
            self._data_factory = self.owl_manager.getOWLDataFactory()
        return self._data_factory

    def create_ontology(self, iri):
        """Creates an empty ontology given a valid IRI string"""

        if not isinstance(iri, str):
            raise TypeError(f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}")
        return self.owl_manager.createOntology(IRI.create(iri))

    def create_class(self, iri):
        """Creates and OWL class given a valid IRI string"""

        if not isinstance(iri, str):
            raise TypeError(f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}")
        return self.data_factory.getOWLClass(IRI.create(iri))

    def create_individual(self, iri):
        """Creates and OWLNamedIndividual given a valid IRI string"""

        if not isinstance(iri, str):
            raise TypeError(f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}")
        return self.data_factory.getOWLNamedIndividual(IRI.create(iri))

    def create_object_property(self, iri):
        """Creates and OWL Object property given a valid IRI string"""

        if not isinstance(iri, str):
            raise TypeError(f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}")
        return self.data_factory.getOWLObjectProperty(IRI.create(iri))

    def create_subclass_of(self, cexpr1, cexpr2):
        """Creates OWLSubClassOfAxiom for a given pair of class expressions"""
        return self.data_factory.getOWLSubClassOfAxiom(cexpr1, cexpr2)

    def create_equivalent_classes(self, *cexprs):
        """Creates OWLEquivalentClassesAxiom for a given list of class expressions"""
        return self.data_factory.getOWLEquivalentClassesAxiom(cexprs)

    def create_disjoint_classes(self, *cexprs):
        """Creates OWLDisjointClassesAxiom for a given list of class expressions"""
        return self.data_factory.getOWLDisjointClassesAxiom(cexprs)

    def create_object_some_values_from(self, obj_prop, cexpr):
        """Creates OWLObjectSomeValuesFrom for a given object property and a class expression"""
        return self.data_factory.getOWLObjectSomeValuesFrom(obj_prop, cexpr)

    def create_object_all_values_from(self, obj_prop, cexpr):
        """Creates OWLObjectAllValuesFrom for a given object property and a class expression"""
        return self.data_factory.getOWLObjectAllValuesFrom(obj_prop, cexpr)

    def create_object_intersection_of(self, *cexprs):
        """Creates OWLObjectIntersectionOf for a given list of class expressions"""
        return self.data_factory.getOWLObjectIntersectionOf(cexprs)

    def create_object_union_of(self, *cexprs):
        """Creates OWLObjectUnionOf for a given list of class expressions"""
        return self.data_factory.getOWLObjectUnionOf(cexprs)

    def create_complement_of(self, cexpr):
        """Creates OWLObjectComplementOf for a given class expression"""
        return self.data_factory.getOWLObjectComplementOf(cexpr)

    def create_class_assertion(self, cexpr, ind):
        """Creates OWLClassAssertionAxiom for a given class expression and individual"""
        return self.data_factory.getOWLClassAssertionAxiom(cexpr, ind)

    def create_object_property_assertion(self, obj_prop, ind1, ind2):
        """Creates OWLObjectPropertyAssertionAxiom for a given object property and two \
individuals"""
        return self.data_factory.getOWLObjectPropertyAssertionAxiom(obj_prop, ind1, ind2)
