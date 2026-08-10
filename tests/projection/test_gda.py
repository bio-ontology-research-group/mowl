from unittest import TestCase

from mowl.owlapi import OWLAPIAdapter
from mowl.projection import GDAProjector, OWL2VecStarProjector
from org.semanticweb.owlapi.model import IRI

import jpype.imports
from java.util import HashSet

OBO = "http://purl.obolibrary.org/obo/"


class TestGDA(TestCase):

    @classmethod
    def setUpClass(self):
        """Builds an ontology with a single two-level nested restriction:

        HP_0000123 SubClassOf has-part some (
            PATO_0000001 and inheres-in some UBERON_0000955)
        """
        adapter = OWLAPIAdapter()
        data_factory = adapter.data_factory
        manager = adapter.owl_manager

        self.ontology = manager.createOntology()

        phenotype = data_factory.getOWLClass(IRI.create(OBO + "HP_0000123"))
        quality = data_factory.getOWLClass(IRI.create(OBO + "PATO_0000001"))
        anatomy = data_factory.getOWLClass(IRI.create(OBO + "UBERON_0000955"))
        has_part = data_factory.getOWLObjectProperty(IRI.create(OBO + "has-part"))
        inheres_in = data_factory.getOWLObjectProperty(IRI.create(OBO + "inheres-in"))

        inner = data_factory.getOWLObjectSomeValuesFrom(inheres_in, anatomy)
        operands = HashSet()
        operands.add(quality)
        operands.add(inner)
        outer = data_factory.getOWLObjectSomeValuesFrom(
            has_part, data_factory.getOWLObjectIntersectionOf(operands))

        manager.addAxiom(self.ontology, data_factory.getOWLSubClassOfAxiom(phenotype, outer))

        self.nested_edge = (OBO + "HP_0000123",
                            "http://mowl.borg/has-part_inheres-in",
                            OBO + "UBERON_0000955")

    def edges(self, projector):
        return {(e.src, e.rel, e.dst) for e in projector.project(self.ontology)}

    def test_constructor_parameter_types(self):
        """This should raise TypeError with message when constructor parameters
        are of incorrect type"""
        self.assertRaisesRegex(
            TypeError,
            "Optional parameter bidirectional_taxonomy must be of type boolean",
            GDAProjector, bidirectional_taxonomy="True")
        self.assertRaisesRegex(
            TypeError, "Optional parameter only_taxonomy must be of type boolean",
            GDAProjector, only_taxonomy="True")
        self.assertRaisesRegex(
            TypeError,
            "Optional parameter include_literals must be of type boolean",
            GDAProjector, include_literals="True")
        self.assertRaisesRegex(
            TypeError,
            "Optional parameter source_prefixes must be an iterable of strings",
            GDAProjector, source_prefixes=OBO + "HP_")
        self.assertRaisesRegex(
            TypeError,
            "Optional parameter target_prefixes must be an iterable of strings",
            GDAProjector, target_prefixes=(1, 2))

    def test_project_method_parameter_types(self):
        """This should raise TypeError with message when project method parameter is of
        incorrect type"""
        projector = GDAProjector()
        for bad_input in ["True", 1, {"a": 1}, None]:
            self.assertRaisesRegex(
                TypeError,
                "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
                projector.project, bad_input)

    def test_owl2vecstar_drops_nested_restriction(self):
        """OWL2VecStar should generate no edge for a non-atomic filler. This is the
        gap GDAProjector fills."""
        self.assertEqual(self.edges(OWL2VecStarProjector()), set())

    def test_project_nested_restriction(self):
        """The nested restriction should be unfolded into a single edge whose
        relation is the composition of the roles traversed"""
        self.assertEqual(self.edges(GDAProjector()), {self.nested_edge})

    def test_only_taxonomy_suppresses_unfolding(self):
        """No unfolding should happen when only taxonomy edges are requested"""
        self.assertEqual(self.edges(GDAProjector(only_taxonomy=True)), set())

    def test_non_matching_source_prefix(self):
        """No edge should be generated when the source class does not match"""
        projector = GDAProjector(source_prefixes=(OBO + "MP_",))
        self.assertEqual(self.edges(projector), set())

    def test_non_matching_target_prefix(self):
        """No edge should be generated when the target class does not match"""
        projector = GDAProjector(target_prefixes=(OBO + "CL_",))
        self.assertEqual(self.edges(projector), set())

    def test_empty_prefixes_match_every_class(self):
        """Empty prefix tuples should disable filtering, exposing the intermediate
        class reached at the first level of the walk"""
        projector = GDAProjector(source_prefixes=(), target_prefixes=())
        self.assertEqual(
            self.edges(projector),
            {self.nested_edge,
             (OBO + "HP_0000123", "http://mowl.borg/has-part", OBO + "PATO_0000001")})
