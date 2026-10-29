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

        # A normalised axiom that cannot be reverse-translated must not be silently dropped
        with self.assertRaisesRegex(RuntimeError, "Reverse translation of normalised axiom"):
            normalizer._ELNormalizer__revert_translation([self.gci0_axiom])

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


class TestElNormalizerCaching(TestCase):
    """Tests for ELNormalizer caching functionality"""

    @classmethod
    def setUpClass(cls):
        cls.family_dataset = FamilyDataset()
        cls.adapter = OWLAPIAdapter()

    def setUp(self):
        """Create a temporary directory for cache files"""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normalize_creates_cache_file(self):
        """Test that normalize creates a cache file when ontology_path is provided"""
        import os

        # Create a test ontology file path
        ontology_path = os.path.join(self.temp_dir, "test_ontology.owl")
        expected_cache_path = os.path.join(self.temp_dir, "test_ontology_mowl_el_normalized.owl")

        normalizer = ELNormalizer()
        normalized_axioms = normalizer.normalize(
            self.family_dataset.ontology,
            ontology_path=ontology_path
        )

        # Check that the cache file was created
        self.assertTrue(os.path.exists(expected_cache_path),
                        f"Cache file should be created at {expected_cache_path}")

        # Verify normalization still works correctly
        self.assertEqual(len(normalized_axioms["gci0"]), 7)
        self.assertEqual(len(normalized_axioms["gci1"]), 2)

    def test_normalize_loads_from_cache(self):
        """Test that normalize loads from cache file when it exists"""
        import os

        ontology_path = os.path.join(self.temp_dir, "test_ontology.owl")
        expected_cache_path = os.path.join(self.temp_dir, "test_ontology_mowl_el_normalized.owl")

        normalizer = ELNormalizer()

        # First call: normalize and create cache
        normalized_axioms_1 = normalizer.normalize(
            self.family_dataset.ontology,
            ontology_path=ontology_path
        )

        self.assertTrue(os.path.exists(expected_cache_path))

        # Second call: should load from cache
        with self.assertLogs(level="INFO") as log:
            normalized_axioms_2 = normalizer.normalize(
                self.family_dataset.ontology,
                ontology_path=ontology_path
            )
            # Check that loading from cache was logged
            log_messages = [record.getMessage() for record in log.records]
            self.assertTrue(
                any("Loading normalized ontology from cache" in msg for msg in log_messages),
                f"Should log loading from cache. Got: {log_messages}"
            )

        # Results should be equivalent
        self.assertEqual(len(normalized_axioms_1["gci0"]), len(normalized_axioms_2["gci0"]))
        self.assertEqual(len(normalized_axioms_1["gci1"]), len(normalized_axioms_2["gci1"]))
        self.assertEqual(len(normalized_axioms_1["gci2"]), len(normalized_axioms_2["gci2"]))
        self.assertEqual(len(normalized_axioms_1["gci3"]), len(normalized_axioms_2["gci3"]))

    def test_normalize_no_cache_without_ontology_path(self):
        """Test that normalize does not create cache when ontology_path is not provided"""
        import os

        normalizer = ELNormalizer()

        # Normalize without ontology_path
        normalized_axioms = normalizer.normalize(self.family_dataset.ontology)

        # No cache files should be created in temp_dir
        cache_files = [f for f in os.listdir(self.temp_dir) if f.endswith("_mowl_el_normalized.owl")]
        self.assertEqual(len(cache_files), 0, "No cache files should be created without ontology_path")

        # Normalization should still work
        self.assertEqual(len(normalized_axioms["gci0"]), 7)

    def test_normalize_use_cache_false_skips_cache(self):
        """Test that normalize skips cache when use_cache=False"""
        import os

        ontology_path = os.path.join(self.temp_dir, "test_ontology.owl")
        expected_cache_path = os.path.join(self.temp_dir, "test_ontology_mowl_el_normalized.owl")

        normalizer = ELNormalizer()

        # First call: normalize and create cache
        normalizer.normalize(
            self.family_dataset.ontology,
            ontology_path=ontology_path
        )
        self.assertTrue(os.path.exists(expected_cache_path))

        # Second call with use_cache=False: should re-normalize
        normalized_axioms = normalizer.normalize(
            self.family_dataset.ontology,
            ontology_path=ontology_path,
            use_cache=False
        )

        # Normalization should still produce correct results
        self.assertEqual(len(normalized_axioms["gci0"]), 7)

    def test_get_cache_path(self):
        """Test the cache path generation logic"""
        import os

        test_cases = [
            ("ontology.owl", "ontology_mowl_el_normalized.owl"),
            ("/path/to/ontology.owl", "/path/to/ontology_mowl_el_normalized.owl"),
            ("my_ontology.owl", "my_ontology_mowl_el_normalized.owl"),
            ("/data/test.owl", "/data/test_mowl_el_normalized.owl"),
        ]

        normalizer = ELNormalizer()
        for input_path, expected_cache_path in test_cases:
            cache_path = normalizer.get_cache_path(input_path)
            self.assertEqual(cache_path, expected_cache_path,
                             f"Cache path for '{input_path}' should be '{expected_cache_path}'")

    def test_normalize_ontology_path_type_checking(self):
        """Test type checking for ontology_path parameter"""
        normalizer = ELNormalizer()

        with self.assertRaisesRegex(TypeError, "Parameter 'ontology_path' must be of type str"):
            normalizer.normalize(self.family_dataset.ontology, ontology_path=123)

    def test_normalize_use_cache_type_checking(self):
        """Test type checking for use_cache parameter"""
        import os

        normalizer = ELNormalizer()
        ontology_path = os.path.join(self.temp_dir, "test.owl")

        with self.assertRaisesRegex(TypeError, "Parameter 'use_cache' must be of type bool"):
            normalizer.normalize(
                self.family_dataset.ontology,
                ontology_path=ontology_path,
                use_cache="yes"
            )


class TestElNormalizerAuxiliaryClasses(TestCase):
    """Regression tests for auxiliary classes introduced by jcel while flattening nested
    class expressions. Auxiliary classes used to receive integer identifiers that collided
    with the identifiers of named classes, so they were reverse-translated into unrelated
    named classes (or the axiom was dropped)."""

    NS = "http://test.org/aux#"

    @classmethod
    def setUpClass(cls):
        cls.adapter = OWLAPIAdapter()
        cls.manager = cls.adapter.owl_manager
        cls.df = cls.adapter.data_factory

    def _cls(self, name):
        return self.adapter.create_class(self.NS + name)

    def _prop(self, name):
        return self.adapter.create_object_property(self.NS + name)

    def _named(self, ce):
        return ce.getClassExpressionType().toString() == "Class"

    def _normalized_owl_axioms(self, normalized):
        return {key: {str(ax.owl_axiom.toString()) for ax in axs}
                for key, axs in normalized.items()
                if key not in ("class_assertion", "object_property_assertion")}

    def _fresh_classes(self, normalized, ontology):
        original = {str(c.toStringID()) for c in ontology.getClassesInSignature()}
        fresh = set()
        for key, axs in normalized.items():
            if key in ("class_assertion", "object_property_assertion"):
                continue
            for ax in axs:
                for c in ax.owl_axiom.getClassesInSignature():
                    if not c.isOWLThing() and not c.isOWLNothing() and \
                       str(c.toStringID()) not in original:
                        fresh.add(str(c.toStringID()))
        return fresh

    def test_nested_existential_in_conjunction(self):
        """B and r some C SubClassOf A, D SubClassOf B normalises to
        r some C SubClassOf X, B and X SubClassOf A, D SubClassOf B (X fresh)."""
        ont = self.adapter.create_ontology("http://test.org/aux/repro")
        A, B, C, D = [self._cls(x) for x in "ABCD"]
        r = self._prop("r")
        self.manager.addAxiom(ont, self.adapter.create_subclass_of(
            self.adapter.create_object_intersection_of(
                B, self.adapter.create_object_some_values_from(r, C)), A))
        self.manager.addAxiom(ont, self.adapter.create_subclass_of(D, B))

        normalized = ELNormalizer().normalize(ont)
        fresh = self._fresh_classes(normalized, ont)
        self.assertEqual(len(fresh), 1, f"Expected exactly one fresh class, got {fresh}")
        X = self.df.getOWLClass(IRI.create(fresh.pop()))

        expected = {
            "gci0": {self.adapter.create_subclass_of(D, B)},
            "gci1": {self.adapter.create_subclass_of(
                self.adapter.create_object_intersection_of(B, X), A)},
            "gci2": set(), "gci0_bot": set(), "gci1_bot": set(), "gci3_bot": set(),
            "gci3": {self.adapter.create_subclass_of(
                self.adapter.create_object_some_values_from(r, C), X)},
        }
        expected = {k: {str(ax.toString()) for ax in v} for k, v in expected.items()}
        self.assertEqual(self._normalized_owl_axioms(normalized), expected)

    def test_cheesey_pizza_definition(self):
        """CheeseyPizza EquivalentTo Pizza and hasTopping some CheeseTopping must not produce
        axioms mentioning unrelated named classes."""
        ont = self.adapter.create_ontology("http://test.org/aux/pizza")
        pizza, cheesey, cheese = self._cls("Pizza"), self._cls("CheeseyPizza"), \
            self._cls("CheeseTopping")
        unrelated = [self._cls(n) for n in ("CajunSpiceTopping", "IceCream", "Unrelated1",
                                           "Unrelated2", "Unrelated3")]
        has_topping = self._prop("hasTopping")
        defn = self.adapter.create_object_intersection_of(
            pizza, self.adapter.create_object_some_values_from(has_topping, cheese))
        self.manager.addAxiom(ont, self.adapter.create_equivalent_classes(cheesey, defn))
        for c in [pizza, cheese] + unrelated:
            self.manager.addAxiom(ont, self.df.getOWLDeclarationAxiom(c))

        normalized = ELNormalizer().normalize(ont)
        fresh = self._fresh_classes(normalized, ont)
        self.assertEqual(len(fresh), 1, f"Expected exactly one fresh class, got {fresh}")
        X = self.df.getOWLClass(IRI.create(fresh.pop()))

        unrelated_iris = {str(c.toStringID()) for c in unrelated}
        for key, axs in normalized.items():
            for ax in axs:
                sig = {str(c.toStringID()) for c in ax.owl_axiom.getClassesInSignature()}
                self.assertFalse(sig & unrelated_iris, f"Unrelated class in {ax.owl_axiom}")

        some = self.adapter.create_object_some_values_from(has_topping, cheese)
        expected = {
            "gci0": {self.adapter.create_subclass_of(cheesey, pizza)},
            "gci1": {self.adapter.create_subclass_of(
                self.adapter.create_object_intersection_of(pizza, X), cheesey)},
            "gci2": {self.adapter.create_subclass_of(cheesey, some)},
            "gci3": {self.adapter.create_subclass_of(some, X)},
            "gci0_bot": set(), "gci1_bot": set(), "gci3_bot": set(),
        }
        expected = {k: {str(ax.toString()) for ax in v} for k, v in expected.items()}
        self.assertEqual(self._normalized_owl_axioms(normalized), expected)

    def _random_concept(self, rng, classes, roles, depth):
        if depth == 0 or rng.random() < 0.3:
            return rng.choice(classes)
        if rng.random() < 0.5:
            n = rng.choice([2, 3])
            operands = [self._random_concept(rng, classes, roles, depth - 1) for _ in range(n)]
            if len({str(o.toString()) for o in operands}) < 2:
                return operands[0]
            return self.adapter.create_object_intersection_of(*operands)
        return self.adapter.create_object_some_values_from(
            rng.choice(roles), self._random_concept(rng, classes, roles, depth - 1))

    def _random_ontology(self, seed):
        import random
        rng = random.Random(seed)
        ont = self.adapter.create_ontology(f"http://test.org/aux/random{seed}")
        n_classes = rng.randint(6, 15)
        n_roles = rng.randint(1, 3)
        classes = [self._cls(f"R{seed}_C{i}") for i in range(n_classes)]
        roles = [self._prop(f"R{seed}_r{i}") for i in range(n_roles)]
        for c in classes:
            self.manager.addAxiom(ont, self.df.getOWLDeclarationAxiom(c))
        for _ in range(rng.randint(8, 20)):
            sub = self._random_concept(rng, classes, roles, 3)
            sup = self._random_concept(rng, classes, roles, 3)
            if rng.random() < 0.2 and self._named(sub):
                axiom = self.adapter.create_equivalent_classes(sub, sup)
            else:
                axiom = self.adapter.create_subclass_of(sub, sup)
            self.manager.addAxiom(ont, axiom)
        for _ in range(rng.randint(0, 2)):
            a, b = rng.sample(classes, 2)
            self.manager.addAxiom(ont, self.adapter.create_disjoint_classes(a, b))
        return ont, classes

    def _entailments(self, ontology, classes):
        from org.semanticweb.elk.owlapi import ElkReasonerFactory
        reasoner = ElkReasonerFactory().createReasoner(ontology)
        try:
            names = [str(c.toStringID()) for c in classes]
            sat = {str(c.toStringID()): bool(reasoner.isSatisfiable(c)) for c in classes}
            subs = set()
            for a in classes:
                a_name = str(a.toStringID())
                if not sat[a_name]:
                    # an unsatisfiable class is subsumed by every class
                    subs |= {(a_name, b) for b in names}
                    continue
                supers = {str(c.toStringID())
                          for c in reasoner.getSuperClasses(a, False).getFlattened()}
                supers |= {str(c.toStringID())
                           for c in reasoner.getEquivalentClasses(a).getEntities()}
                subs |= {(a_name, b) for b in names if b in supers or b == a_name}
        finally:
            reasoner.dispose()
        return sat, subs

    def test_random_nested_ontologies_conservative_extension(self):
        """Normalisation must be a conservative extension over the original signature:
        satisfiability and subsumption between original named classes agree."""
        for seed in range(20):
            with self.subTest(seed=seed):
                ont, classes = self._random_ontology(seed)
                normalized = ELNormalizer().normalize(ont)

                norm_ont = self.adapter.create_ontology(f"http://test.org/aux/norm{seed}")
                for c in classes:
                    self.manager.addAxiom(norm_ont, self.df.getOWLDeclarationAxiom(c))
                for key, axs in normalized.items():
                    if key in ("class_assertion", "object_property_assertion"):
                        continue
                    for ax in axs:
                        self.manager.addAxiom(norm_ont, ax.owl_axiom)

                sat_o, subs_o = self._entailments(ont, classes)
                sat_n, subs_n = self._entailments(norm_ont, classes)
                self.assertEqual(sat_o, sat_n)
                self.assertEqual(subs_o, subs_n)
