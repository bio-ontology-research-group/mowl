from tests.datasetFactory import FamilyDataset, PPIYeastSlimDataset
from unittest import TestCase
import os
import shutil
import mowl
mowl.init_jvm("10g")
from mowl.corpus import extract_and_save_axiom_corpus, \
    extract_and_save_annotation_corpus, extract_axiom_corpus, extract_annotation_corpus


class TestBase(TestCase):

    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()
        self.ppi_yeast_slim_dataset = PPIYeastSlimDataset()

    def test_extract_and_save_axiom_corpus_params_types(self):
        """This should test the type checking of the method `extract_and_save_axiom_corpus`."""

        out_file = "/tmp/out_file.txt"

        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            extract_and_save_axiom_corpus, "ontology", "out_file")

        self.assertRaisesRegex(TypeError, "Parameter out_file must be of type str",
                               extract_and_save_axiom_corpus, self.family_dataset.ontology, 1)

        self.assertRaisesRegex(
            TypeError, "Optional parameter mode must be of type str",
            extract_and_save_axiom_corpus, self.family_dataset.ontology, out_file, 1)

    def test_extract_and_save_axiom_corpus_value_mode(self):
        """This should test the value of the attribute mode to be correct in the method \
`extract_and_save_axiom_corpus`."""

        out_file = "/tmp/out_file.txt"
        self.assertRaisesRegex(
            ValueError, "Parameter mode must be a file reading mode. Options are 'a' or 'w'",
            extract_and_save_axiom_corpus, self.family_dataset.ontology, out_file, mode="alkdfal")

    def test_extract_and_save_axiom_corpus_out_file_exists(self):
        """This should test the existence of the output file of the method \
`extract_and_save_axiom_corpus`."""

        out_file = "/tmp/out_file.txt"
        extract_and_save_axiom_corpus(self.family_dataset.ontology, out_file)
        self.assertTrue(os.path.exists(out_file))
        os.remove(out_file)

    def test_extract_and_save_axiom_corpus_write_and_append_mode(self):
        """This should test the write and append mode of the method \
`extract_and_save_axiom_corpus`."""

        out_file = "/tmp/out_file.txt"
        extract_and_save_axiom_corpus(self.family_dataset.ontology, out_file, "w")
        with open(out_file, "r") as f:
            content_first = f.readlines()

        extract_and_save_axiom_corpus(self.family_dataset.ontology, out_file, "w")
        with open(out_file, "r") as f:
            content_second = f.readlines()

        self.assertEqual(content_first, content_second)

        extract_and_save_axiom_corpus(self.family_dataset.ontology, out_file, "a")
        with open(out_file, "r") as f:
            content_third = f.readlines()

        self.assertEqual(content_first + content_second, content_third)
        os.remove(out_file)
    # 3

    def test_extract_and_save_annotation_corpus_params_types(self):
        """This should test the type checking of the method \
`extract_and_save_annotation_corpus`."""

        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            extract_and_save_annotation_corpus, "ontology", "out_file")

        self.assertRaisesRegex(TypeError, "Parameter out_file must be of type str",
                               extract_and_save_annotation_corpus, self.family_dataset.ontology, 1)

        self.assertRaisesRegex(
            TypeError, "Optional parameter mode must be of type str",
            extract_and_save_annotation_corpus, self.family_dataset.ontology, "out_file", 1)

    def test_extract_and_save_annotation_corpus_value_mode(self):
        """This should test the value of the attribute mode to be correct in the method \
`extract_and_save_annotation_corpus`."""

        out_file = "/tmp/out_file.txt"
        self.assertRaisesRegex(
            ValueError,
            "Parameter mode must be a file reading mode. Options are 'a' or 'w'",
            extract_and_save_annotation_corpus, self.family_dataset.ontology,
            out_file, mode="alkdfal")

    def test_extract_and_save_annotation_corpus_out_file_exists(self):
        """This should test the existence of the output file of the method \
`extract_and_save_annotation_corpus`."""

        out_file = "/tmp/out_file2.txt"
        extract_and_save_annotation_corpus(self.family_dataset.ontology, out_file)
        self.assertTrue(os.path.exists(out_file))
        os.remove(out_file)

    def test_extract_and_save_annotation_corpus_write_and_append_mode(self):
        """This should test the write and append mode of the method \
`extract_and_save_annotation_corpus`."""

        out_file = "/tmp/out_file2.txt"
        extract_and_save_annotation_corpus(self.ppi_yeast_slim_dataset.ontology, out_file, "w")
        with open(out_file, "r") as f:
            content_first = f.readlines()

        extract_and_save_annotation_corpus(self.ppi_yeast_slim_dataset.ontology, out_file, "w")
        with open(out_file, "r") as f:
            content_second = f.readlines()

        self.assertEqual(content_first, content_second)

        extract_and_save_annotation_corpus(self.ppi_yeast_slim_dataset.ontology, out_file, "a")
        with open(out_file, "r") as f:
            content_third = f.readlines()

        self.assertEqual(content_first + content_second, content_third)

        os.remove(out_file)

    #######################################

    def test_extract_axiom_corpus_params_types(self):
        """This should test the type checking of the method `extract_axiom_corpus`."""

        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            extract_axiom_corpus, "ontology")

    def test_extract_axiom_corpus_return_type(self):
        """This should test the return type of the method `extract_axiom_corpus`."""

        self.assertIsInstance(extract_axiom_corpus(self.family_dataset.ontology), list)

    #########################################

    def test_extract_annotation_corpus_params_types(self):
        """This should test the type checking of the method `extract_annotation_corpus`."""

        self.assertRaisesRegex(
            TypeError,
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology",
            extract_annotation_corpus, "ontology")

    def test_extract_annotation_corpus_return_type(self):
        """This should test the return type of the method `extract_annotation_corpus`."""

        self.assertIsInstance(extract_annotation_corpus(self.ppi_yeast_slim_dataset.ontology),
                              list)
