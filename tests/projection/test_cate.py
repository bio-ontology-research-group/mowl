from tests.datasetFactory import FamilyDataset
from mowl.projection import CategoricalProjector
from mowl.owlapi.defaults import TOP, BOT
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.model import IRI
from unittest import TestCase

import mowl.error.messages as msg

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
        output_type = 1
        self.assertRaisesRegex(
            TypeError, msg.type_error("output_type", "str", type(output_type)),
            CategoricalProjector, output_type)

        saturation_steps = "1"
        self.assertRaisesRegex(
            TypeError, msg.type_error("saturation_steps", "int", type(saturation_steps), optional=True),
            CategoricalProjector, "str", saturation_steps=saturation_steps)

        self.assertRaisesRegex(
            ValueError, "Optional parameter saturation_steps must be non-negative",
            CategoricalProjector, "str", saturation_steps = -1)

        transitive_closure = "True"
        self.assertRaisesRegex(
            TypeError, msg.type_error("transitive_closure", "bool", type(transitive_closure), optional=True),
            CategoricalProjector, "str", transitive_closure=transitive_closure)

        def_6 = 1
        self.assertRaisesRegex(
            TypeError, msg.type_error("def_6", "bool", type(def_6), optional=True),
            CategoricalProjector, "str", def_6=def_6)

        def_7 = "True"
        self.assertRaisesRegex(
            TypeError, msg.type_error("def_7", "bool", type(def_7), optional=True),
            CategoricalProjector, "str", def_7=def_7)

        lemma_6 = 1
        self.assertRaisesRegex(
            TypeError, msg.type_error("lemma_6", "bool", type(lemma_6), optional=True),
            CategoricalProjector, "str", lemma_6=lemma_6)

        lemma_8 = "True"
        self.assertRaisesRegex(
            TypeError, msg.type_error("lemma_8", "bool", type(lemma_8), optional=True),
            CategoricalProjector, "str", lemma_8=lemma_8)
        
        
    def test_project_method_parameter_types(self):
        """This should check if the project method parameters are of the correct type"""
        projector = CategoricalProjector("str")
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, "True")

    def test_project_family_ontology(self):
        """This should check if the projection result is correct"""
        projector = CategoricalProjector("str")
        edges = projector.project(self.ontology)
        edges = set([e.astuple() for e in edges])
        edges = set([get_edge(e[0], e[2]) for e in edges])

        ground_truth_edges = set()
        with open("tests/projection/fixtures/cate_family.csv") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue
                line = line.strip()
                sub, super_ = line.split(",")
                ground_truth_edges.add(get_edge(sub, super_))
        
        self.assertEqual(set(edges), ground_truth_edges)

def get_edge(a, b):
    return (a, "http://arrow", b)
