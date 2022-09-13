import mowl
from tests.datasetFactory import PPIYeastDataset, PPIYeastSlimDataset, GDAHumanDataset, \
    GDAHumanELDataset, GDAMouseDataset, GDAMouseELDataset, FamilyDataset
from unittest import TestCase
import os
import shutil


class TestInstanceOfDataset(TestCase):

    def test_family_is_instance_of_dataset(self):
        """This should check if FamilyDataset is an instance of Dataset"""
        dataset = FamilyDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_ppi_yeast_is_instance_of_dataset(self):
        """This should check if PPIYeastDataset is an instance of Dataset"""
        dataset = PPIYeastDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_ppi_yeast_slim_is_instance_of_dataset(self):
        """This should check if PPIYeastSlimDataset is an instance of Dataset"""
        dataset = PPIYeastSlimDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_gda_human_is_instance_of_dataset(self):
        """This should check if GDAHumanDataset is an instance of Dataset"""
        dataset = GDAHumanDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_gda_human_el_is_instance_of_dataset(self):
        """This should check if GDAHumanELDataset is an instance of Dataset"""
        dataset = GDAHumanELDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_gda_mouse_is_instance_of_dataset(self):
        """This should check if GDAMouseDataset is an instance of Dataset"""
        dataset = GDAMouseDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)

    def test_gda_mouse_el_is_instance_of_dataset(self):
        """This should check if GDAMouseELDataset is an instance of Dataset"""
        dataset = GDAMouseELDataset()
        self.assertIsInstance(dataset, mowl.datasets.Dataset)
