import mowl
from tests.datasetFactory import PPIYeastSlimDataset, GDAHumanELDataset, GDAMouseELDataset, \
    FamilyDataset
from unittest import TestCase
import os
import shutil


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
