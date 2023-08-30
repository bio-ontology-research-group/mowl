from mowl.projection import OWL2VecStarProjector
from unicodedata import bidirectional
from unittest import TestCase
from mowl.owlapi.defaults import TOP
from mowl.owlapi import OWLAPIAdapter
from tests.datasetFactory import FamilyDataset
from org.semanticweb.owlapi.model import IRI

import jpype.imports
from java.util import HashSet, ArrayList

# TODO: Rules 3 and 6 of projection table missing


class TestOwl2VecStar(TestCase):

    @classmethod
    def setUpClass(self):
        dataset = FamilyDataset()
        self.ontology = dataset.ontology

    def test_constructor_parameter_types(self):
        """This should raise TypeError with message when constructor parameter
        are of incorrect type"""
        self.assertRaisesRegex(
            TypeError,
            "Optional parameter bidirectional_taxonomy must be of type boolean",
            OWL2VecStarProjector, bidirectional_taxonomy="True")
        self.assertRaisesRegex(
            TypeError, "Optional parameter only_taxonomy must be of type boolean",
            OWL2VecStarProjector, only_taxonomy="True")
        self.assertRaisesRegex(
            TypeError,
            "Optional parameter include_literals must be of type boolean",
            OWL2VecStarProjector, include_literals="True")

    def test_project_method_parameter_types(self):
        """This should raise TypeError with message when project method parameter is of
        incorrect type"""
        projector = OWL2VecStarProjector()
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, "True")
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, 1)
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, {"a": 1, "b": 2, "c": 3})
        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            projector.project, None)

    def test_project_family_ontology(self):
        """This should check if the projection result is correct"""
        adapter = OWLAPIAdapter()
        data_factory = adapter.data_factory
        ont_manager = adapter.owl_manager

        extra_triples = set()

        class1 = data_factory.getOWLClass(IRI.create("http://class1"))
        class2 = data_factory.getOWLClass(IRI.create("http://class2"))
        role1 = data_factory.getOWLObjectProperty(IRI.create("http://role1"))
        role2 = data_factory.getOWLObjectProperty(IRI.create("http://role2"))
        # top_class = data_factory.getOWLClass(IRI.create(TOP))

        # Rule 2
        axiom_rule_2_1 = data_factory.getOWLObjectPropertyDomainAxiom(role1, class1)
        axiom_rule_2_2 = data_factory.getOWLObjectPropertyRangeAxiom(role1, class2)
        triple_rule2 = ("http://class1", "http://role1", "http://class2")
        # To fulfill rule 5
        extra_triples.add(("http://class2", "http://inverseRole1", "http://class1"))

        # Rule 3
        # named_individual1 = data_factory.getOWLNamedIndividual(IRI.create("http://individual1"))
        # exists_some_individual1 = data_factory.getOWLObjectSomeValuesFrom(role2,
        # named_individual1.asOWLClass())
        # axiom_rule_3_1 = data_factory.getOWLSubClassOfAxiom(class1, exists_some_individual1)
        # axiom_rule_3_2 = data_factory.getOWLAssertionAxiom(class2, named_individual1)
        # triple_rule3 = ("http://class1", "http://role2", "http://class2")

        # Rule 4
        sub_role1 = data_factory.getOWLObjectProperty(IRI.create("http://subrole1"))
        axiom_rule_4 = data_factory.getOWLSubObjectPropertyOfAxiom(sub_role1, role1)
        triple_rule4 = ("http://class1", "http://subrole1", "http://class2")

        # Rule 5
        inverse_role1 = data_factory.getOWLObjectProperty(IRI.create("http://inverseRole1"))
        axiom_rule_5 = data_factory.getOWLInverseObjectPropertiesAxiom(role1, inverse_role1)
        triple_rule5 = ("http://class2", "http://inverseRole1", "http://class1")

        # Rule 6
        class10 = data_factory.getOWLClass(IRI.create("http://class10"))
        class11 = data_factory.getOWLClass(IRI.create("http://class11"))
        class12 = data_factory.getOWLClass(IRI.create("http://class12"))

        exists_some_class11 = data_factory.getOWLObjectSomeValuesFrom(role1, class11)
        axiom_rule_6_1 = data_factory.getOWLSubClassOfAxiom(class10, exists_some_class11)
        exists_some_class12 = data_factory.getOWLObjectSomeValuesFrom(role2, class12)
        axiom_rule_6_2 = data_factory.getOWLSubClassOfAxiom(class11, exists_some_class12)

        prop_chain = IRI.create("http://superpropertyChain")
        superproperty_chain = data_factory.getOWLObjectProperty(prop_chain)
        property_chain = ArrayList()
        property_chain.add(role1)
        property_chain.add(role2)
        axiom_rule_6_3 = data_factory.getOWLSubPropertyChainOfAxiom(property_chain,
                                                                    superproperty_chain)

        triple_rule6_1 = ("http://class10", "http://role1", "http://class11")
        triple_rule6_2 = ("http://class11", "http://role2", "http://class12")
        # triple_rule6_3 = ("http://class10", "http://superpropertyChain", "http://class12")
        # To fulfill rule 4
        extra_triples.add(("http://class10", "http://subrole1", "http://class11"))
        # To fulfill rule 5
        extra_triples.add(("http://class11", "http://inverseRole1", "http://class10"))

        # Rule 8
        individual = data_factory.getOWLNamedIndividual(IRI.create("http://individual"))
        axiom_rule_8 = data_factory.getOWLClassAssertionAxiom(class1, individual)
        triple_rule8 = ("http://individual", "http://type", "http://class1")

        # Rule 9
        individual2 = data_factory.getOWLNamedIndividual(IRI.create("http://individual2"))
        axiom_rule_9 = data_factory.getOWLObjectPropertyAssertionAxiom(role1,
                                                                       individual,
                                                                       individual2)
        triple_rule9 = ("http://individual", "http://role1", "http://individual2")

        projector = OWL2VecStarProjector()
        ontology = self.ontology

        ont_manager.addAxiom(ontology, axiom_rule_2_1)
        ont_manager.addAxiom(ontology, axiom_rule_2_2)

        ont_manager.addAxiom(ontology, axiom_rule_4)
        ont_manager.addAxiom(ontology, axiom_rule_5)
        ont_manager.addAxiom(ontology, axiom_rule_6_1)
        ont_manager.addAxiom(ontology, axiom_rule_6_2)
        ont_manager.addAxiom(ontology, axiom_rule_6_3)
        ont_manager.addAxiom(ontology, axiom_rule_8)
        ont_manager.addAxiom(ontology, axiom_rule_9)

        edges = projector.project(ontology)
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
        ground_truth_edges.add(("http://Parent", "http://hasChild", "http://Person"))

        ground_truth_edges.add(triple_rule2)
        # ground_truth_edges.add(triple_rule3)
        ground_truth_edges.add(triple_rule4)
        ground_truth_edges.add(triple_rule5)
        ground_truth_edges.add(triple_rule6_1)
        ground_truth_edges.add(triple_rule6_2)
        # ground_truth_edges.add(triple_rule6_3)
        ground_truth_edges.add(triple_rule8)
        ground_truth_edges.add(triple_rule9)

        ground_truth_edges = ground_truth_edges.union(extra_triples)

        self.assertEqual(set(edges), ground_truth_edges)
