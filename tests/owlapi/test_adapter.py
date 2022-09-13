from org.semanticweb.owlapi.model import OWLDataFactory, OWLClass, OWLOntologyManager
from org.semanticweb.owlapi.apibinding import OWLManager
from mowl.owlapi import OWLAPIAdapter
from unittest import TestCase
import mowl.error as err


class TestAdapter(TestCase):

    @classmethod
    def setUpClass(self):
        owlapi_owl_manager = OWLManager.createOWLOntologyManager()
        self.owlapi_data_factory = owlapi_owl_manager.getOWLDataFactory()

        self.iri1 = "http://mowl/iri1"

    def test_create_owl_manager(self):
        """Create an OWLManager from OWLAPI"""
        adapter = OWLAPIAdapter()
        owl_manager = adapter.owl_manager
        self.assertIsInstance(owl_manager, OWLOntologyManager)

    def test_create_data_factory(self):
        """Create an OWLDataFactory from OWLAPI"""
        adapter = OWLAPIAdapter()
        data_factory = adapter.data_factory
        self.assertIsInstance(data_factory, OWLDataFactory)

    def test_create_owl_class_from_iri_string(self):
        """Create an OWLClass from string IRI"""
        adapter = OWLAPIAdapter()
        owlclass = adapter.create_class(self.iri1)
        self.assertIsInstance(owlclass, OWLClass)

        # Sad paths
        self.assertRaisesRegex(
            TypeError, f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}",
            adapter.create_class, 1)
        self.assertRaisesRegex(
            TypeError, f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}",
            adapter.create_class, True)
        self.assertRaisesRegex(
            TypeError, f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}",
            adapter.create_class, lambda: 1)
