from unittest import TestCase
from mowl.ontology.normalize import ELNormalizer, GCI, GCI0, GCI1, GCI2, GCI3, GCI0_BOT, \
    GCI1_BOT, GCI3_BOT, process_axiom
from tests.datasetFactory import FamilyDataset
from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import BOT

from uk.ac.manchester.cs.owl.owlapi import SWRLRuleImpl, OWLEquivalentObjectPropertiesAxiomImpl

from org.semanticweb.owlapi.model import IRI
from java.util import HashSet


class TestElNormalizer(TestCase):

    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()

        self.adapter = OWLAPIAdapter()
        self.data_factory = self.adapter.data_factory
        self.bot = self.adapter.create_class(BOT)
        # GCI0 Axioms
        subclass = self.adapter.create_class("http://class1")
        superclass = self.adapter.create_class("http://class2")
        self.gci0_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)
        self.gci0_bot_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, self.bot)

        # GCI1 Axiom
        subclass2 = self.adapter.create_class("http://class3")
        intersection = self.data_factory.getOWLObjectIntersectionOf(subclass, subclass2)
        self.gci1_axiom = self.data_factory.getOWLSubClassOfAxiom(intersection, superclass)
        self.gci1_bot_axiom = self.data_factory.getOWLSubClassOfAxiom(intersection, self.bot)

        # GCI2 Axiom
        role = self.data_factory.getOWLObjectProperty(IRI.create("http://role"))
        some = self.data_factory.getOWLObjectSomeValuesFrom(role, superclass)
        self.gci2_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, some)

        # GCI3 Axiom
        self.gci3_axiom = self.data_factory.getOWLSubClassOfAxiom(some, subclass)
        self.gci3_bot_axiom = self.data_factory.getOWLSubClassOfAxiom(some, self.bot)

        # UnionOf Axiom
        superclass = self.data_factory.getOWLObjectUnionOf(subclass, subclass2)
        self.union_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)
        self.union_axiom_inverted = self.data_factory.getOWLSubClassOfAxiom(superclass, subclass)

        # MinCardinality Axiom
        superclass = self.data_factory.getOWLObjectMinCardinality(2, role, superclass)
        self.min_cardinality_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)

        # ComplementOf Axiom
        superclass = self.data_factory.getOWLObjectComplementOf(superclass)
        self.complement_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)

        # AllValuesFrom Axiom
        superclass = self.data_factory.getOWLObjectAllValuesFrom(role, superclass)
        self.all_values_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)

        # MaxCardinality Axiom
        superclass = self.data_factory.getOWLObjectMaxCardinality(2, role, superclass)
        self.max_cardinality_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)

        # ExactCardinality Axiom
        superclass = self.data_factory.getOWLObjectExactCardinality(2, role, superclass)
        self.exact_cardinality_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass,
                                                                               superclass)

        # Annotation Axiom
        annot_prop = self.data_factory.getOWLAnnotationProperty(IRI.create("http://annotation"))
        class1 = IRI.create("http://class1")
        literal = self.data_factory.getOWLLiteral("test")
        self.annotation_axiom = self.data_factory.getOWLAnnotationAssertionAxiom(annot_prop,
                                                                                 class1, literal)

        # ObjectHasSelf Axiom
        has_self = self.data_factory.getOWLObjectHasSelf(role)
        self.object_has_self_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, has_self)

        # urn:swrl rule
        set1 = HashSet()
        variable = self.data_factory.getSWRLVariable(IRI.create("http://var"))
        set1.add(self.data_factory.getSWRLClassAtom(subclass, variable))

        set2 = HashSet()
        variable = self.data_factory.getSWRLVariable(IRI.create("http://var"))
        set2.add(self.data_factory.getSWRLClassAtom(superclass, variable))

        self.swrl_rule = SWRLRuleImpl(set1, set2)

        # EquivalentObjectProperties Axiom
        role_set = HashSet()
        role_set.add(role)
        role2 = self.data_factory.getOWLObjectProperty(IRI.create("http://role2"))
        role_set.add(role2)

        equivalent = OWLEquivalentObjectPropertiesAxiomImpl(role_set, HashSet())
        self.equivalent_object_properties_axiom = equivalent

        # SymmetricObjectProperty Axiom
        symmetric = self.data_factory.getOWLSymmetricObjectPropertyAxiom(role)
        self.symmetric_object_property_axiom = symmetric

        # AsymmetricObjectProperty Axiom
        asymmetric = self.data_factory.getOWLAsymmetricObjectPropertyAxiom(role)
        self.asymmetric_object_property_axiom = asymmetric

        # ObjectOneOf Axiom
        individual = self.data_factory.getOWLNamedIndividual(IRI.create("http://individual"))
        superclass = self.data_factory.getOWLObjectOneOf(individual)
        self.object_one_of_axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)

        # New ontology
        owl_manager = self.adapter.owl_manager
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
org.semanticweb.owlapi.model.OWLOntology. Found: <class 'str'>"):
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

        # Test _revert_translations method
        with self.assertLogs(level="INFO") as log:
            normalizer._ELNormalizer__revert_translation([self.gci0_axiom])
            message = f"Reverse translation. Ignoring axiom: {self.gci0_axiom}"
            self.assertEqual(log.records[0].getMessage(), message)

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

        self.assertEqual(process_axiom(self.gci0_bot_axiom),
                         ("gci0_bot", GCI0_BOT(self.gci0_bot_axiom)))
        self.assertEqual(process_axiom(self.gci1_bot_axiom),
                         ("gci1_bot", GCI1_BOT(self.gci1_bot_axiom)))
        self.assertEqual(process_axiom(self.gci3_bot_axiom),
                         ("gci3_bot", GCI3_BOT(self.gci3_bot_axiom)))

    def test_process_axiom_logs(self):
        """This checks for the logs produced by process_axiom method"""

        with self.assertLogs(level="INFO") as log:
            process_axiom(self.union_axiom)
            message = f"Superclass type not recognized. Ignoring axiom: {self.union_axiom}"
            self.assertEqual(log.records[0].getMessage(), message)

        with self.assertLogs(level="INFO") as log:
            process_axiom(self.union_axiom_inverted)
            message = f"Subclass type not recognized. Ignoring axiom: {self.union_axiom_inverted}"
            self.assertEqual(log.records[0].getMessage(), message)
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

        self.assertEqual(self.ontology.getAxiomCount(), 13)
        ontology = normalizer.preprocess_ontology(self.ontology)

        self.assertEqual(ontology.getAxiomCount(), 0)

    # Test GCIs

    def test_gci0(self):
        """"This should check the correct behaviour of GCI0"""
        axiom = GCI0(self.gci0_axiom)
        # Attributes
        self.assertEqual(axiom.subclass, "http://class1")
        self.assertEqual(axiom.superclass, "http://class2")

        # Flattened entities
        classes = {"http://class1", "http://class2"}
        roles = set()
        inds = set()
        self.assertEqual(axiom.get_entities(), (classes, roles, inds))

    def test_gci1(self):
        """"This should check the correct behaviour of GCI1"""
        axiom = GCI1(self.gci1_axiom)
        # Attributes
        self.assertEqual(axiom.left_subclass, "http://class1")

        # Restart axiom to improve coverage
        axiom = GCI1(self.gci1_axiom)
        self.assertEqual(axiom.right_subclass, "http://class3")
        self.assertEqual(axiom.superclass, "http://class2")

        # Flattened entities
        classes = {"http://class1", "http://class2", "http://class3"}
        roles = set()
        inds = set()
        self.assertEqual(axiom.get_entities(), (classes, roles, inds))

    def test_gci2(self):
        """"This should check the correct behaviour of GCI2"""
        axiom = GCI2(self.gci2_axiom)
        # Attributes
        self.assertEqual(axiom.subclass, "http://class1")
        self.assertEqual(axiom.object_property, "http://role")

        # Restart axiom
        axiom = GCI2(self.gci2_axiom)
        self.assertEqual(axiom.filler, "http://class2")

        # Flattened entities
        classes = {"http://class1", "http://class2"}
        roles = {"http://role"}
        inds = set()
        self.assertEqual(axiom.get_entities(), (classes, roles, inds))

    def test_gci3(self):
        """"This should check the correct behaviour of GCI3"""
        axiom = GCI3(self.gci3_axiom)
        # Attributes
        self.assertEqual(axiom.object_property, "http://role")

        # Restart axiom
        axiom = GCI3(self.gci3_axiom)
        self.assertEqual(axiom.filler, "http://class2")
        self.assertEqual(axiom.superclass, "http://class1")

        # Flattened entities
        classes = {"http://class1", "http://class2"}
        roles = {"http://role"}
        inds = set()
        self.assertEqual(axiom.get_entities(), (classes, roles, inds))

    # Test bot axioms

    def test_gci0_bot(self):
        """This should check the correct behaviour of GCI0_BOT"""

        subclass = self.adapter.create_class("http://class1")
        axiom_good = GCI0_BOT(self.data_factory.getOWLSubClassOfAxiom(subclass, self.bot))
        self.assertIsInstance(axiom_good, GCI0)

        with self.assertRaisesRegex(ValueError,
                                    "Superclass in GCI0_BOT must be the bottom concept."):
            GCI0_BOT(self.gci0_axiom)

    def test_gci1_bot(self):
        """This should check the correct behaviour of GCI1_BOT"""

        class1 = self.data_factory.getOWLClass(IRI.create("http://class1"))
        class2 = self.data_factory.getOWLClass(IRI.create("http://class2"))
        intersection = self.data_factory.getOWLObjectIntersectionOf(class1, class2)
        axiom_good = GCI1_BOT(self.data_factory.getOWLSubClassOfAxiom(intersection, self.bot))

        self.assertIsInstance(axiom_good, GCI1)

        with self.assertRaisesRegex(ValueError,
                                    "Superclass in GCI1_BOT must be the bottom concept."):
            GCI1_BOT(self.gci1_axiom)

    def test_gci3_bot(self):
        """This should check the correct behaviour of GCI3_BOT"""

        class1 = self.data_factory.getOWLClass(IRI.create("http://class1"))
        role = self.data_factory.getOWLObjectProperty(IRI.create("http://role"))
        some = self.data_factory.getOWLObjectSomeValuesFrom(role, class1)

        axiom_good = GCI3_BOT(self.data_factory.getOWLSubClassOfAxiom(some, self.bot))

        self.assertIsInstance(axiom_good, GCI3)

        with self.assertRaisesRegex(ValueError,
                                    "Superclass in GCI3_BOT must be the bottom concept."):
            GCI3_BOT(self.gci3_axiom)

    def test_static_method_get_entities(self):
        """This should check the correct behaviour of the static method get_entities"""

        # Attributes
        gci = GCI(self.gci0_axiom)
        self.assertEqual(gci.owl_axiom, self.gci0_axiom)

        classes, roles, inds = GCI.get_entities([GCI0(self.gci0_axiom), GCI1(self.gci1_axiom),
                                           GCI2(self.gci2_axiom), GCI3(self.gci3_axiom)])

        self.assertEqual(classes, {"http://class1", "http://class2", "http://class3"})
        self.assertEqual(roles, {"http://role"})
        self.assertEqual(inds, set())
