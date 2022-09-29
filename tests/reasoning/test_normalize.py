from unittest import TestCase
from mowl.reasoning.normalize import ELNormalizer, GCI0, GCI1, GCI2, GCI3, process_axiom
from mowl.datasets.builtin import FamilyDataset
from mowl.owlapi import OWLAPIAdapter

from uk.ac.manchester.cs.owl.owlapi import SWRLRuleImpl, OWLEquivalentObjectPropertiesAxiomImpl

from org.semanticweb.owlapi.model import IRI
from java.util import HashSet   

class TestElNormalizer(TestCase):
    
    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()
        
        adapter = OWLAPIAdapter()
        data_factory = adapter.data_factory
        # GCI0 Axiom
        subclass = adapter.create_class("http://class1")
        superclass = adapter.create_class("http://class2")
        self.gci0_axiom = data_factory.getOWLSubClassOfAxiom(subclass, superclass)

        # GCI1 Axiom
        subclass2 = adapter.create_class("http://class3")
        intersection = data_factory.getOWLObjectIntersectionOf(subclass, subclass2)
        self.gci1_axiom = data_factory.getOWLSubClassOfAxiom(intersection, superclass)

        # GCI2 Axiom
        role = data_factory.getOWLObjectProperty(IRI.create("http://role"))
        some = data_factory.getOWLObjectSomeValuesFrom(role, superclass)
        self.gci2_axiom = data_factory.getOWLSubClassOfAxiom(subclass, some)

        # GCI3 Axiom
        self.gci3_axiom = data_factory.getOWLSubClassOfAxiom(some, subclass)

        # UnionOf Axiom
        self.union_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectUnionOf(subclass, subclass2))

        # MinCardinality Axiom
        self.min_cardinality_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectMinCardinality(2, role, superclass))

        # ComplementOf Axiom
        self.complement_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectComplementOf(superclass))

        # AllValuesFrom Axiom
        self.all_values_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectAllValuesFrom(role, superclass))

        # MaxCardinality Axiom
        self.max_cardinality_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectMaxCardinality(2, role, superclass))

        # ExactCardinality Axiom
        self.exact_cardinality_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectExactCardinality(2, role, superclass))

        # Annotation Axiom
        self.annotation_axiom = data_factory.getOWLAnnotationAssertionAxiom(data_factory.getOWLAnnotationProperty(IRI.create("http://annotation")), IRI.create("http://class1"), data_factory.getOWLLiteral("test"))

        # ObjectHasSelf Axiom
        self.object_has_self_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectHasSelf(role))

        # urn:swrl rule
        set1 = HashSet()
        set1.add(data_factory.getSWRLClassAtom(subclass, data_factory.getSWRLVariable(IRI.create("http://var"))))
        
        set2 = HashSet()
        set2.add(data_factory.getSWRLClassAtom(superclass, data_factory.getSWRLVariable(IRI.create("http://var"))))

        self.swrl_rule = SWRLRuleImpl(set1, set2)

        # EquivalentObjectProperties Axiom
        role_set = HashSet()
        role_set.add(role)
        role2 = data_factory.getOWLObjectProperty(IRI.create("http://role2"))
        role_set.add(role2)

        self.equivalent_object_properties_axiom = OWLEquivalentObjectPropertiesAxiomImpl(role_set, HashSet())

        # SymmetricObjectProperty Axiom
        self.symmetric_object_property_axiom = data_factory.getOWLSymmetricObjectPropertyAxiom(role)

        # AsymmetricObjectProperty Axiom
        self.asymmetric_object_property_axiom = data_factory.getOWLAsymmetricObjectPropertyAxiom(role)

        # ObjectOneOf Axiom
        self.object_one_of_axiom = data_factory.getOWLSubClassOfAxiom(subclass, data_factory.getOWLObjectOneOf(data_factory.getOWLNamedIndividual(IRI.create("http://individual"))))

        # New ontology
        owl_manager = adapter.owl_manager
        self.ontology = owl_manager.createOntology(IRI.create("http://ontology"))
        self.ontology.addAxiom(self.union_axiom)
        self.ontology.addAxiom(self.min_cardinality_axiom)
        self.ontology.addAxiom(self.complement_axiom)
        self.ontology.addAxiom(self.all_values_axiom)
        self.ontology.addAxiom(self.max_cardinality_axiom)
        self.ontology.addAxiom(self.exact_cardinality_axiom)
        self.ontology.addAxiom(self.annotation_axiom)
        self.ontology.addAxiom(self.object_has_self_axiom)
        self.ontology.addAxiom(self.swrl_rule)
        self.ontology.addAxiom(self.equivalent_object_properties_axiom)
        self.ontology.addAxiom(self.symmetric_object_property_axiom)
        self.ontology.addAxiom(self.asymmetric_object_property_axiom)
        self.ontology.addAxiom(self.object_one_of_axiom)

    # Test normalize method

    def test_normalize_type_checking(self):
        """This performs type checking on the normalize method"""

        normalizer = ELNormalizer()
        with self.assertRaisesRegex(TypeError, "Parameter 'ontology' must be of type \
org.semanticweb.owlapi.model.OWLOntology"):
            normalizer.normalize("test")

    def test_normalize(self):
        """This checks the correct result of EL normalization over FamilyDataset"""

        normalizer = ELNormalizer()
        normalized_axioms = normalizer.normalize(self.family_dataset.ontology)

        self.assertEqual(len(normalized_axioms["gci0"]), 7)
        self.assertEqual(len(normalized_axioms["gci1"]), 2)
        self.assertEqual(len(normalized_axioms["gci2"]), 1)
        self.assertEqual(len(normalized_axioms["gci3"]), 1)
        self.assertEqual(len(normalized_axioms["gci0_bot"]), 0)
        self.assertEqual(len(normalized_axioms["gci1_bot"]), 1)
        self.assertEqual(len(normalized_axioms["gci3_bot"]), 0)

    # Test process_axiom method

    def test_process_axiom_type_checking(self):
        """This performs type checking on the process_axiom method"""

        with self.assertRaisesRegex(TypeError, "Parameter 'axiom' must be of type \
org.semanticweb.owlapi.model.OWLAxiom"):
            process_axiom("test")

    def test_process_axiom(self):
        """This checks the correct result of process_axiom method"""


        self.assertEqual(process_axiom(self.gci0_axiom), ("gci0", GCI0(self.gci0_axiom)))
        self.assertEqual(process_axiom(self.gci1_axiom), ("gci1", GCI1(self.gci1_axiom)))
        self.assertEqual(process_axiom(self.gci2_axiom), ("gci2", GCI2(self.gci2_axiom)))
        self.assertEqual(process_axiom(self.gci3_axiom), ("gci3", GCI3(self.gci3_axiom)))

    # Test preprocess_ontology method

    def test_preprocess_ontology_type_checking(self):
        """This performs type checking on the preprocess_ontology method"""

        normalizer = ELNormalizer()
        with self.assertRaisesRegex(TypeError, "Parameter 'ontology' must be of type \
org.semanticweb.owlapi.model.OWLOntology"):
            normalizer.preprocess_ontology("test")

    def test_preprocess_ontology(self):
        """This checks the correct behaviour of preprocess_ontology method"""
        
        normalizer = ELNormalizer()

        ontology = normalizer.preprocess_ontology(self.ontology)

        self.assertEqual(ontology.getAxiomCount(), 0)
