import mowl
from tests.datasetFactory import PPIYeastSlimDataset, GDAHumanELDataset, GDAMouseELDataset, FamilyDataset, GOSubsumptionDataset, FoodOnSubsumptionDataset
from unittest import TestCase
import os
import shutil
import random

from org.semanticweb.owlapi.model import OWLClass


class TestInstanceOfDataset(TestCase):

    def test_family_is_instance_of_dataset(self):
        """This should check if FamilyDataset is an instance of Dataset"""
        dataset = FamilyDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_ppi_yeast_slim_is_instance_of_dataset(self):
        """This should check if PPIYeastSlimDataset is an instance of Dataset"""
        dataset = PPIYeastSlimDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_gda_human_el_is_instance_of_dataset(self):
        """This should check if GDAHumanELDataset is an instance of Dataset"""
        dataset = GDAHumanELDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_gda_mouse_el_is_instance_of_dataset(self):
        """This should check if GDAMouseELDataset is an instance of Dataset"""
        dataset = GDAMouseELDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_go_subsumption_is_instance_of_dataset(self):
        """This should check if GOSubsumptionDataset is an instance of Dataset"""
        dataset = GOSubsumptionDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_foodon_subsumption_is_instance_of_dataset(self):
        """This should check if FoodOnSubsumptionDataset is an instance of Dataset"""
        dataset = FoodOnSubsumptionDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

        
    def test_evaluation_classes_ppi(self):
        """This should check the correct behaviour of evaluation_classes_method in ppi dataset"""
        dataset = PPIYeastSlimDataset()

        str_classes = dataset.evaluation_classes[0].as_str
        owl_classes = dataset.evaluation_classes[0].as_owl
        dict_classes = dataset.evaluation_classes[0].as_dict

        self.assertEqual(str_classes, list(dict_classes.keys()))
        self.assertEqual(owl_classes, list(dict_classes.values()))

        rand_index = random.randint(0, len(str_classes) - 1)
        self.assertIsInstance(str_classes[rand_index], str)
        self.assertIsInstance(owl_classes[rand_index], OWLClass)

    def test_evaluation_classes_gda_human(self):
        """This should check the correct behaviour of evaluation_classes_method in gda_human \
dataset"""
        dataset = GDAHumanELDataset()

        classes_genes, classes_diseases = dataset.evaluation_classes
        str_classes_genes = classes_genes.as_str
        owl_classes_genes = classes_genes.as_owl

        str_classes_diseases = classes_diseases.as_str
        owl_classes_diseases = classes_diseases.as_owl

        dict_classes_genes = classes_genes.as_dict
        dict_classes_diseases = classes_diseases.as_dict

        self.assertEqual(str_classes_genes, list(dict_classes_genes.keys()))
        self.assertEqual(owl_classes_genes, list(dict_classes_genes.values()))

        self.assertEqual(str_classes_diseases, list(dict_classes_diseases.keys()))
        self.assertEqual(owl_classes_diseases, list(dict_classes_diseases.values()))

        rand_index = random.randint(0, len(str_classes_genes) - 1)
        self.assertIsInstance(str_classes_genes[rand_index], str)
        self.assertIsInstance(owl_classes_genes[rand_index], OWLClass)

        rand_index = random.randint(0, len(str_classes_diseases) - 1)
        self.assertIsInstance(str_classes_diseases[rand_index], str)
        self.assertIsInstance(owl_classes_diseases[rand_index], OWLClass)


