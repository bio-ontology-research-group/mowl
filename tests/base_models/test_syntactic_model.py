from mowl.base_models import SyntacticModel
from tests.datasetFactory import FamilyDataset
from mowl.owlapi import OWLAPIAdapter
from unittest import TestCase
import  mowl.error.messages as msg
from org.semanticweb.owlapi.model import AddAxiom
from org.semanticweb.owlapi.model.parameters import Imports
import os

class TestSyntacticModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = FamilyDataset()
        adapter = OWLAPIAdapter()
        manager = adapter.owl_manager
        factory = adapter.data_factory
        #import org.semanticweb.owlapi.model.*;
        father = adapter.create_class("http://Father")

        annotation_property = factory.getRDFSLabel();
        annotation_value = factory.getOWLLiteral("This is the father class.")
        annotation = factory.getOWLAnnotation(annotation_property, annotation_value);

        axiom = factory.getOWLAnnotationAssertionAxiom(father.getIRI(), annotation);
        manager.applyChange(AddAxiom(self.dataset.ontology, axiom));

        self.imports = Imports.fromBoolean(True)

        
    def test_parameter_type(self):
        """This should test the type of the parameters"""

        with self.assertRaisesRegex(TypeError, "Optional parameter 'corpus_filepath' must be of type str."):
            SyntacticModel(self.dataset, corpus_filepath=1)

    def test_get_corpus_filepath_attribute(self):
        """This checks if Syntactic.corpus_filepath attribute works correctly"""

        model = SyntacticModel(self.dataset, corpus_filepath="test")
        corpus_filepath = model.corpus_filepath
        self.assertIsInstance(corpus_filepath, str)
        self.assertEqual(corpus_filepath, "test")

    
    def test_temporary_corpus_filepath(self):
        """This checks temporary corpus filepath"""

        model = SyntacticModel(self.dataset)
        corpus_filepath = model.corpus_filepath
        self.assertIsInstance(corpus_filepath, str)
        import tempfile
        tmppath = tempfile.gettempdir()
        self.assertTrue(corpus_filepath.startswith(tmppath))

    def test_corpus_not_generated(self):
        """This should test the correct behaviour when the corpus is not generated"""

        model = SyntacticModel(self.dataset)
        with self.assertRaisesRegex(AttributeError, msg.CORPUS_NOT_GENERATED):
            model.corpus

    

    def test_generate_corpus_with_save(self):
        """This should test the generate_corpus method with save=True"""

        model = SyntacticModel(self.dataset)
        model.generate_corpus()
        self.assertTrue(os.path.exists(model.corpus_filepath))
        with open(model.corpus_filepath, "r") as f:
            axiom_corpus = f.readlines()

        self.assertEqual(len(axiom_corpus), len(self.dataset.ontology.getAxioms(self.imports)))

        model.generate_corpus(with_annotations=True)
        self.assertTrue(os.path.exists(model.corpus_filepath))
        with open(model.corpus_filepath, "r") as f:
            full_corpus = f.readlines()

        self.assertEqual(len(full_corpus), len(self.dataset.ontology.getAxioms(self.imports)) + 1)


    def test_generate_corpus_no_save(self):
        """This should test the generate_corpus method with save=False"""

        model = SyntacticModel(self.dataset)
        axiom_corpus = model.generate_corpus(save=False)

        self.assertEqual(len(axiom_corpus), len(self.dataset.ontology.getAxioms(self.imports)))

        full_corpus = model.generate_corpus(save=False, with_annotations=True)
        self.assertEqual(len(full_corpus), len(self.dataset.ontology.getAxioms(self.imports)) + 1)
