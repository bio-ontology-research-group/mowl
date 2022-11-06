from tests.datasetFactory import FamilyDataset
from mowl.projection import TaxonomyProjector
from mowl.owlapi.defaults import TOP
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.model import IRI
from unittest import TestCase


class TestTaxonomy(TestCase):

    @classmethod
    def setUpClass(self):
        dataset = FamilyDataset()
        self.ontology = dataset.ontology

    def test_constructor_parameter_types(self):
        """This should check if the constructor parameters are of the correct type"""
        self.assertRaisesRegex(
            TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean",
            TaxonomyProjector, "True")

    def test_project_method_parameter_types(self):
        """This should check if the project method parameters are of the correct type"""
        projector = TaxonomyProjector()
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, "True")

    def test_project_family_ontology(self):
        """This should check if the projection result is correct"""
        projector = TaxonomyProjector()
        edges = projector.project(self.ontology)
        edges = set([e.astuple() for e in edges])

        ground_truth_edges = set()
        ground_truth_edges.add(("http://Male", "http://subclassof", "http://Person"))
        ground_truth_edges.add(("http://Female", "http://subclassof", "http://Person"))
        ground_truth_edges.add(("http://Father", "http://subclassof", "http://Male"))
        ground_truth_edges.add(("http://Mother", "http://subclassof", "http://Female"))
        ground_truth_edges.add(("http://Father", "http://subclassof", "http://Parent"))
        ground_truth_edges.add(("http://Mother", "http://subclassof", "http://Parent"))
        ground_truth_edges.add(("http://Parent", "http://subclassof", "http://Person"))

        self.assertEqual(set(edges), ground_truth_edges)

    def test_project_family_bidirectional_taxonomy(self):
        """This should check if bidirectional taxonomy projection is correct"""
        projector = TaxonomyProjector(bidirectional_taxonomy=True)
        edges = projector.project(self.ontology)
        edges = set([e.astuple() for e in edges])

        ground_truth_edges = set()
        ground_truth_edges.add(("http://Male", "http://subclassof", "http://Person"))
        ground_truth_edges.add(("http://Female", "http://subclassof", "http://Person"))
        ground_truth_edges.add(("http://Father", "http://subclassof", "http://Male"))
        ground_truth_edges.add(("http://Mother", "http://subclassof", "http://Female"))
        ground_truth_edges.add(("http://Father", "http://subclassof", "http://Parent"))
        ground_truth_edges.add(("http://Mother", "http://subclassof", "http://Parent"))
        ground_truth_edges.add(("http://Parent", "http://subclassof", "http://Person"))

        ground_truth_edges.add(("http://Person", "http://superclassof", "http://Male"))
        ground_truth_edges.add(("http://Person", "http://superclassof", "http://Female"))
        ground_truth_edges.add(("http://Male", "http://superclassof", "http://Father"))
        ground_truth_edges.add(("http://Female", "http://superclassof", "http://Mother"))
        ground_truth_edges.add(("http://Parent", "http://superclassof", "http://Father"))
        ground_truth_edges.add(("http://Parent", "http://superclassof", "http://Mother"))
        ground_truth_edges.add(("http://Person", "http://superclassof", "http://Parent"))

        self.assertEqual(set(edges), ground_truth_edges)
