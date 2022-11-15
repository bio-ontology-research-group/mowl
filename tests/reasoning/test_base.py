from org.semanticweb.HermiT import Reasoner
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from mowl.reasoning import MOWLReasoner
from tests.datasetFactory import FamilyDataset
from org.semanticweb.owlapi.model import OWLSubClassOfAxiom, OWLEquivalentClassesAxiom, \
    OWLDisjointClassesAxiom
from random import randrange
from unittest import TestCase


class TestMowlReasoner(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()

    def test_attribute_type_checking(self):
        """This should test if type checking is applied for MOWLReasoner class"""

        with self.assertRaisesRegex(
            TypeError,
            "Parameter reasoner must be an instance of \
org.semanticweb.owlapi.reasoner.OWLReasoner"):
            _ = MOWLReasoner("reasoner")

###############################################
    def test_infer_subclass_axioms_type_checking(self):
        """This should test if type checking is applied for infer_subclass_axioms method"""
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
        reasoner.precomputeInferences()

        mowl_reasoner = MOWLReasoner(reasoner)
        with self.assertRaisesRegex(TypeError, "All elements in parameter owl_classes must be of \
type org.semanticweb.owlapi.model.OWLClass"):
            mowl_reasoner.infer_subclass_axioms(["owl:Thing"])

        with self.assertRaisesRegex(TypeError, "Optional parameter direct must be of type bool"):
            mowl_reasoner.infer_subclass_axioms([], direct="True")

    def test_return_values_infer_subclass_axioms_method(self):
        """This should test if the return values of infer_subclass_axioms method are correct"""
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
        reasoner.precomputeInferences()

        mowl_reasoner = MOWLReasoner(reasoner)
        classes = self.dataset.ontology.getClassesInSignature()
        result = mowl_reasoner.infer_subclass_axioms(classes)

        self.assertIsInstance(result, list)
        rand_idx = randrange(0, len(result))
        self.assertIsInstance(result[rand_idx], OWLSubClassOfAxiom)

    def test_parameter_direct(self):
        """This should test if the parameter direct is working correctly"""
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
        reasoner.precomputeInferences()

        mowl_reasoner = MOWLReasoner(reasoner)
        classes = self.dataset.ontology.getClassesInSignature()
        result_direct = mowl_reasoner.infer_subclass_axioms(classes, direct=True)
        result_not_direct = mowl_reasoner.infer_subclass_axioms(classes, direct=False)
        assert len(result_direct) < len(result_not_direct)
        ###############################################

    def test_infer_equivalent_class_axioms_type_checking(self):
        """This should test if type checking is applied for infer_equivalent_class_axioms method"""
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
        reasoner.precomputeInferences()

        mowl_reasoner = MOWLReasoner(reasoner)
        with self.assertRaisesRegex(TypeError, "All elements in parameter owl_classes must be of \
type org.semanticweb.owlapi.model.OWLClass"):
            mowl_reasoner.infer_equivalent_class_axioms(["ontology"])

    def test_return_values_infer_equivalent_class_axioms_method(self):
        """This should test if the return values of infer_equivalent_class_axioms method are \
correct"""
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
        reasoner.precomputeInferences()

        mowl_reasoner = MOWLReasoner(reasoner)
        classes = self.dataset.ontology.getClassesInSignature()
        result = mowl_reasoner.infer_equivalent_class_axioms(classes)

        self.assertIsInstance(result, list)
        rand_idx = randrange(0, len(result))
        self.assertIsInstance(result[rand_idx], OWLEquivalentClassesAxiom)

###############################################

    def test_infer_disjoint_class_axioms_type_checking(self):
        """This should test if type checking is applied for infer_disjoint_class_axioms method"""
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(self.dataset.ontology)
        reasoner.precomputeInferences()

        mowl_reasoner = MOWLReasoner(reasoner)
        with self.assertRaisesRegex(
            TypeError,
            "All elements in parameter owl_classes must be of type \
org.semanticweb.owlapi.model.OWLClass"):
            mowl_reasoner.infer_disjoint_class_axioms(["ontology"])

    def test_return_values_infer_disjoint_class_axioms_method(self):
        """This should test if the return values of infer_disjoint_class_axioms method are \
correct"""

        hermit_reasoner = Reasoner.ReasonerFactory().createReasoner(self.dataset.ontology)
        hermit_reasoner.precomputeInferences()

        mowl_reasoner = MOWLReasoner(hermit_reasoner)
        classes = self.dataset.ontology.getClassesInSignature()
        result = mowl_reasoner.infer_disjoint_class_axioms(classes)

        self.assertIsInstance(result, list)
        rand_idx = randrange(0, len(result))
        self.assertIsInstance(result[rand_idx], OWLDisjointClassesAxiom)
