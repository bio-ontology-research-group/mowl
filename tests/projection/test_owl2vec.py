from unicodedata import bidirectional
from unittest import TestCase
import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector

class TestOwl2VecStar(TestCase):

    def test_constructor_parameter_types(self):
        """This should raise TypeError with message when constructor parameter are of incorrect type"""
        self.assertRaisesRegex(TypeError, "Optional parameter bidirectional_taxonomy must be of type boolean", OWL2VecStarProjector, bidirectional_taxonomy = "True")
        self.assertRaisesRegex(TypeError, "Optional parameter only_taxonomy must be of type boolean", OWL2VecStarProjector, only_taxonomy = "True")
        self.assertRaisesRegex(TypeError, "Optional parameter include_literals must be of type boolean", OWL2VecStarProjector, include_literals = "True")
    
    def test_project_method_parameter_types(self):
        """This should raise TypeError with message when project method parameter is of incorrect type"""
        projector = OWL2VecStarProjector()
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, "True")
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, 1)
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, {"a": 1, "b": 2, "c": 3})
        self.assertRaisesRegex(TypeError, "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology", projector.project, None)

    #TODO: Add test to check if projection result is correct. Do this by comparing with original OWL2VecStar.