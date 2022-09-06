from unittest import TestCase
import mowl
mowl.init_jvm("10g")
import shutil
import os
from mowl.owlapi.defaults import TOP
from mowl.projection import DL2VecProjector
from mowl.datasets.builtin import FamilyDataset

class TestDl2Vec(TestCase):

    @classmethod
    def tearDownClass(self):
        os.remove("family.tar.gz")
        shutil.rmtree("family")

    def test_constructor_parameter_types(self):
        """This should check if the constructor parameters are of the correct type"""
        self.assertRaisesRegex(TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean", DL2VecProjector, "True")
        self.assertRaisesRegex(TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean", DL2VecProjector, 1)
        self.assertRaisesRegex(TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean", DL2VecProjector, {"a": 1, "b": 2, "c": 3})
        self.assertRaisesRegex(TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean", DL2VecProjector, None)

    def test_project_method_parameter_types(self):
        """This should check if the project method parameters are of the correct type"""
        projector = DL2VecProjector()
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, "True")
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, 1)
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, {"a": 1, "b": 2, "c": 3})
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, None)

    #TODO: Add test to check if projection result is correct. To start, do this with Family ontology.

    def test_project_family_ontology(self):
        """This should check if the projection result is correct"""
        ds = FamilyDataset()
        projector = DL2VecProjector()
        edges = projector.project(ds.ontology)
        edges = set([e.astuple() for e in edges])

        ground_truth_edges = set()
        ground_truth_edges.add(("http://Male", "subclassOf", "http://Person"))
        ground_truth_edges.add(("http://Female", "subclassOf", "http://Person"))
        ground_truth_edges.add(("http://Father", "subclassOf", "http://Male"))
        ground_truth_edges.add(("http://Mother", "subclassOf", "http://Female"))
        ground_truth_edges.add(("http://Father", "subclassOf", "http://Parent"))
        ground_truth_edges.add(("http://Mother", "subclassOf", "http://Parent"))
        ground_truth_edges.add(("http://Parent", "subclassOf", "http://Person"))
        ground_truth_edges.add(("http://Parent", "http://hasChild", TOP))

        self.assertEqual(set(edges), ground_truth_edges)