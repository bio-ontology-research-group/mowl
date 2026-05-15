from org.semanticweb.owlapi.model import (
    OWLDataFactory,
    OWLClass,
    OWLOntologyManager,
    OWLOntology,
    OWLNamedIndividual,
    OWLObjectProperty,
    OWLSubClassOfAxiom,
    OWLEquivalentClassesAxiom,
    OWLDisjointClassesAxiom,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectIntersectionOf,
    OWLObjectUnionOf,
    OWLObjectComplementOf,
    OWLClassAssertionAxiom,
    OWLObjectPropertyAssertionAxiom,
)
from org.semanticweb.owlapi.apibinding import OWLManager
from mowl.owlapi import OWLAPIAdapter
from unittest import TestCase
import mowl.error as err


class TestAdapter(TestCase):

    @classmethod
    def setUpClass(cls):
        owlapi_owl_manager = OWLManager.createOWLOntologyManager()
        cls.owlapi_data_factory = owlapi_owl_manager.getOWLDataFactory()

        cls.iri1 = "http://mowl/iri1"
        cls.iri2 = "http://mowl/iri2"
        cls.iri_prop = "http://mowl/prop1"
        cls.iri_ont = "http://mowl/ontology"

        adapter = OWLAPIAdapter()
        cls.cls1 = adapter.create_class(cls.iri1)
        cls.cls2 = adapter.create_class(cls.iri2)
        cls.prop = adapter.create_object_property(cls.iri_prop)
        cls.ind1 = adapter.create_individual(cls.iri1)
        cls.ind2 = adapter.create_individual(cls.iri2)

    def _sad_paths(self, method):
        """Assert that non-string IRI arguments raise TypeError."""
        adapter = OWLAPIAdapter()
        for bad in (1, True, lambda: 1):
            self.assertRaisesRegex(
                TypeError,
                f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}",
                method.__get__(adapter),
                bad,
            )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    def test_create_owl_manager(self):
        """Create an OWLManager from OWLAPI"""
        adapter = OWLAPIAdapter()
        self.assertIsInstance(adapter.owl_manager, OWLOntologyManager)

    def test_create_data_factory(self):
        """Create an OWLDataFactory from OWLAPI"""
        adapter = OWLAPIAdapter()
        self.assertIsInstance(adapter.data_factory, OWLDataFactory)

    # ------------------------------------------------------------------ #
    # IRI-based factory methods (with type validation)
    # ------------------------------------------------------------------ #

    def test_create_owl_class_from_iri_string(self):
        """Create an OWLClass from string IRI"""
        adapter = OWLAPIAdapter()
        self.assertIsInstance(adapter.create_class(self.iri1), OWLClass)
        self._sad_paths(OWLAPIAdapter.create_class)

    def test_create_ontology(self):
        """Create an OWLOntology from string IRI"""
        adapter = OWLAPIAdapter()
        self.assertIsInstance(adapter.create_ontology(self.iri_ont), OWLOntology)
        self._sad_paths(OWLAPIAdapter.create_ontology)

    def test_create_individual(self):
        """Create an OWLNamedIndividual from string IRI"""
        adapter = OWLAPIAdapter()
        self.assertIsInstance(adapter.create_individual(self.iri1), OWLNamedIndividual)
        self._sad_paths(OWLAPIAdapter.create_individual)

    def test_create_object_property(self):
        """Create an OWLObjectProperty from string IRI"""
        adapter = OWLAPIAdapter()
        self.assertIsInstance(adapter.create_object_property(self.iri_prop), OWLObjectProperty)
        self._sad_paths(OWLAPIAdapter.create_object_property)

    # ------------------------------------------------------------------ #
    # Axiom factory methods
    # ------------------------------------------------------------------ #

    def test_create_subclass_of(self):
        """Create an OWLSubClassOfAxiom"""
        adapter = OWLAPIAdapter()
        axiom = adapter.create_subclass_of(self.cls1, self.cls2)
        self.assertIsInstance(axiom, OWLSubClassOfAxiom)

    def test_create_equivalent_classes(self):
        """Create an OWLEquivalentClassesAxiom"""
        adapter = OWLAPIAdapter()
        axiom = adapter.create_equivalent_classes(self.cls1, self.cls2)
        self.assertIsInstance(axiom, OWLEquivalentClassesAxiom)

    def test_create_disjoint_classes(self):
        """Create an OWLDisjointClassesAxiom"""
        adapter = OWLAPIAdapter()
        axiom = adapter.create_disjoint_classes(self.cls1, self.cls2)
        self.assertIsInstance(axiom, OWLDisjointClassesAxiom)

    def test_create_class_assertion(self):
        """Create an OWLClassAssertionAxiom"""
        adapter = OWLAPIAdapter()
        axiom = adapter.create_class_assertion(self.cls1, self.ind1)
        self.assertIsInstance(axiom, OWLClassAssertionAxiom)

    def test_create_object_property_assertion(self):
        """Create an OWLObjectPropertyAssertionAxiom"""
        adapter = OWLAPIAdapter()
        axiom = adapter.create_object_property_assertion(self.prop, self.ind1, self.ind2)
        self.assertIsInstance(axiom, OWLObjectPropertyAssertionAxiom)

    # ------------------------------------------------------------------ #
    # Class-expression factory methods
    # ------------------------------------------------------------------ #

    def test_create_object_some_values_from(self):
        """Create an OWLObjectSomeValuesFrom restriction"""
        adapter = OWLAPIAdapter()
        expr = adapter.create_object_some_values_from(self.prop, self.cls1)
        self.assertIsInstance(expr, OWLObjectSomeValuesFrom)

    def test_create_object_all_values_from(self):
        """Create an OWLObjectAllValuesFrom restriction"""
        adapter = OWLAPIAdapter()
        expr = adapter.create_object_all_values_from(self.prop, self.cls1)
        self.assertIsInstance(expr, OWLObjectAllValuesFrom)

    def test_create_object_intersection_of(self):
        """Create an OWLObjectIntersectionOf expression"""
        adapter = OWLAPIAdapter()
        expr = adapter.create_object_intersection_of(self.cls1, self.cls2)
        self.assertIsInstance(expr, OWLObjectIntersectionOf)

    def test_create_object_union_of(self):
        """Create an OWLObjectUnionOf expression"""
        adapter = OWLAPIAdapter()
        expr = adapter.create_object_union_of(self.cls1, self.cls2)
        self.assertIsInstance(expr, OWLObjectUnionOf)

    def test_create_complement_of(self):
        """Create an OWLObjectComplementOf expression"""
        adapter = OWLAPIAdapter()
        expr = adapter.create_complement_of(self.cls1)
        self.assertIsInstance(expr, OWLObjectComplementOf)
