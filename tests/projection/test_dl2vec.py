from tests.datasetFactory import FamilyDataset
from mowl.projection import DL2VecProjector
from mowl.owlapi.defaults import TOP
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.model import IRI
from unittest import TestCase


class TestDl2Vec(TestCase):

    @classmethod
    def setUpClass(self):
        dataset = FamilyDataset()
        self.ontology = dataset.ontology

    def test_constructor_parameter_types(self):
        """This should check if the constructor parameters are of the correct type"""
        self.assertRaisesRegex(
            TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean",
            DL2VecProjector, "True")
        self.assertRaisesRegex(
            TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean",
            DL2VecProjector, 1)
        self.assertRaisesRegex(
            TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean",
            DL2VecProjector, {"a": 1, "b": 2, "c": 3})
        self.assertRaisesRegex(
            TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean",
            DL2VecProjector, None)

    def test_project_method_parameter_types(self):
        """This should check if the project method parameters are of the correct type"""
        projector = DL2VecProjector()
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, "True")

        self.assertRaisesRegex(
            TypeError,
            "Optional parameter with_individuals must be of type boolean",
            projector.project, self.ontology, with_individuals="True")

        self.assertRaisesRegex(
            TypeError,
            "Optional parameter verbose must be of type boolean",
            projector.project, self.ontology, verbose="True")

    def test_project_family_ontology(self):
        """This should check if the projection result is correct"""
        projector = DL2VecProjector()
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
        ground_truth_edges.add(("http://Parent", "http://hasChild", TOP))

        self.assertEqual(set(edges), ground_truth_edges)

    def test_project_method_with_individuals(self):
        """This check the projection result with 'with_individuals' parameter set to True"""
        projector = DL2VecProjector()

        ontology = self.ontology
        adapter = OWLAPIAdapter()
        data_factory = adapter.data_factory
        ont_manager = adapter.owl_manager

        # Individual, relation, concept
        named_individual = data_factory.getOWLNamedIndividual(IRI.create("http://John"))
        class1 = data_factory.getOWLClass(IRI.create("http://Male"))
        role = data_factory.getOWLObjectProperty(IRI.create("http://hasChild"))
        expression = data_factory.getOWLObjectSomeValuesFrom(role, class1)
        axiom = data_factory.getOWLClassAssertionAxiom(expression, named_individual)

        # Individual, relation, individual
        named_individual2 = data_factory.getOWLNamedIndividual(IRI.create("http://Mary"))
        axiom2 = data_factory.getOWLObjectPropertyAssertionAxiom(role,
                                                                 named_individual,
                                                                 named_individual2)

        ont_manager.addAxiom(ontology, axiom)
        ont_manager.addAxiom(ontology, axiom2)
        edges = projector.project(ontology, with_individuals=True, verbose=True)
        edges = {e.astuple() for e in edges}

        tuple_of_interest = ("http://John", "http://hasChild", "http://Male")
        self.assertIn(tuple_of_interest, edges)

        tuple_of_interest = ("http://John", "http://hasChild", "http://Mary")
        self.assertIn(tuple_of_interest, edges)
