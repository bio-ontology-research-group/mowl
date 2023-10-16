from tests.datasetFactory import FamilyDataset
from mowl.projection import CategoricalProjector
from mowl.owlapi.defaults import TOP, BOT
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.model import IRI
from unittest import TestCase


class TestCat(TestCase):

    @classmethod
    def setUpClass(self):
        dataset = FamilyDataset()
        self.ontology = dataset.ontology

        
        adapter = OWLAPIAdapter()
        data_factory = adapter.data_factory
        ont_manager = adapter.owl_manager

        ind1 = data_factory.getOWLNamedIndividual(IRI.create("http://John"))
        ind2 = data_factory.getOWLNamedIndividual(IRI.create("http://Mary"))
        ind3 = data_factory.getOWLNamedIndividual(IRI.create("http://Peter"))
        role = data_factory.getOWLObjectProperty(IRI.create("http://hasChild"))
        class1 = data_factory.getOWLClass(IRI.create("http://Male"))

        expression = data_factory.getOWLObjectSomeValuesFrom(role, class1)
        axiom = data_factory.getOWLClassAssertionAxiom(expression, ind1)

        # Individual, relation, individual
        
        axiom2 = data_factory.getOWLObjectPropertyAssertionAxiom(role,
                                                                 ind1,
                                                                 ind2)
        axiom3 = data_factory.getOWLClassAssertionAxiom(class1, ind3)

        
        ont_manager.addAxiom(self.ontology, axiom)
        ont_manager.addAxiom(self.ontology, axiom2)
        ont_manager.addAxiom(self.ontology, axiom3)



        
    def test_constructor_parameter_types(self):
        """This should check if the constructor parameters are of the correct type"""
        self.assertRaisesRegex(
            TypeError, "Optional parameter saturation_steps must be of type int",
            CategoricalProjector, "0")
        self.assertRaisesRegex(
            ValueError, "Optional parameter saturation_steps must be non-negative",
            CategoricalProjector, -1)
        self.assertRaisesRegex(
            TypeError, "Optional parameter transitive_closure must be of type bool",
            CategoricalProjector, 1, {"a"})
        
        
    def test_project_method_parameter_types(self):
        """This should check if the project method parameters are of the correct type"""
        projector = CategoricalProjector()
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, "True")

    def test_project_family_ontology(self):
        """This should check if the projection result is correct"""
        projector = CategoricalProjector()
        edges = projector.project(self.ontology)
        edges = set([e.astuple() for e in edges])

        ground_truth_edges = set()
        with open("tests/projection/fixtures/cate_family.csv") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue
                line = line.strip()
                sub, super_ = line.split(",")
                ground_truth_edges.add(edge(sub, super_))
        
        self.assertEqual(set(edges), ground_truth_edges)

def edge(a, b):
    return (a, "http://arrow", b)
