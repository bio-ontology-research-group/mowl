from unittest import TestCase
import mowl
mowl.init_jvm("10g")

from mowl.owlapi import OWLOntology, OWLClass
from mowl.owlapi.defaults import BOT, TOP
import org.semanticweb.owlapi.model as J

class TestObjects(TestCase):

    @classmethod
    def setUpClass(self):
        self.bot_str = "http://www.w3.org/2002/07/owl#Nothing"
        self.top_str = "http://www.w3.org/2002/07/owl#Thing"
        
    def test_types(self):
        """Types returned from mowl.owlapi are OWLAPI types"""
        self.assertEqual(OWLOntology, J.OWLOntology)
        self.assertEqual(OWLClass, J.OWLClass)

    def test_defaults_top_and_bottom(self):
        """This checks if defaults TOP and BOT are correct"""
        self.assertEqual(BOT, self.bot_str)
        self.assertEqual(TOP, self.top_str)
