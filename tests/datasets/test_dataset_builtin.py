from unittest import TestCase
import os
import mowl
import shutil
mowl.init_jvm("10g")

from tests.datasetFactory import PPIYeastDataset, PPIYeastSlimDataset, GDAHumanDataset, GDAHumanELDataset, GDAMouseDataset, GDAMouseELDataset, FamilyDataset




class DeprecatedTestDownloadDatasets(TestCase):

    @classmethod
    def tearDownClass(self):
        os.remove('../ppi_yeast.tar.gz')
        os.remove('../ppi_yeast_slim.tar.gz')
        os.remove('../gda_human.tar.gz')
        os.remove('../gda_human_el.tar.gz')
        os.remove('../gda_mouse.tar.gz')
        os.remove('../gda_mouse_el.tar.gz')
        os.remove('../family.tar.gz')
        shutil.rmtree('../ppi_yeast')
        shutil.rmtree('../ppi_yeast_slim')
        shutil.rmtree('../gda_human')
        shutil.rmtree('../gda_human_el')
        shutil.rmtree('../gda_mouse')
        shutil.rmtree('../gda_mouse_el')
        shutil.rmtree('../family')

    def test_download_ppi_yeast_dataset(self):
        """This should download and check paths of the PPIYeastDataset"""
        dataset = PPIYeastDataset()
        self.assertTrue(os.path.exists('ppi_yeast/ontology.owl'))
        self.assertTrue(os.path.exists('ppi_yeast/valid.owl'))
        self.assertTrue(os.path.exists('ppi_yeast/test.owl'))

    def test_download_ppi_yeast_slim_dataset(self):
        """This should download and check paths of the PPIYeastSlimDataset"""
        dataset = PPIYeastSlimDataset()
        self.assertTrue(os.path.exists('ppi_yeast_slim/ontology.owl'))
        self.assertTrue(os.path.exists('ppi_yeast_slim/valid.owl'))
        self.assertTrue(os.path.exists('ppi_yeast_slim/test.owl'))

    def test_download_gda_human_dataset(self):
        """This should download and check paths of the GDAHumanDataset"""
        dataset = GDAHumanDataset()
        self.assertTrue(os.path.exists('gda_human/ontology.owl'))
        self.assertTrue(os.path.exists('gda_human/valid.owl'))
        self.assertTrue(os.path.exists('gda_human/test.owl'))

    def test_download_gda_human_el_dataset(self):
        """This should download and check paths of the GDAHumanELDataset"""
        dataset = GDAHumanELDataset()
        self.assertTrue(os.path.exists('gda_human_el/ontology.owl'))
        self.assertTrue(os.path.exists('gda_human_el/valid.owl'))
        self.assertTrue(os.path.exists('gda_human_el/test.owl'))

    def test_download_gda_mouse_dataset(self):
        """This should download and check paths of the GDAMouseDataset"""
        dataset = GDAMouseDataset()
        self.assertTrue(os.path.exists('gda_mouse/ontology.owl'))
        self.assertTrue(os.path.exists('gda_mouse/valid.owl'))
        self.assertTrue(os.path.exists('gda_mouse/test.owl'))

    def test_download_gda_mouse_el_dataset(self):
        """This should download and check paths of the GDAMouseELDataset"""
        dataset = GDAMouseELDataset()
        self.assertTrue(os.path.exists('gda_mouse_el/ontology.owl'))
        self.assertTrue(os.path.exists('gda_mouse_el/valid.owl'))
        self.assertTrue(os.path.exists('gda_mouse_el/test.owl'))

    def test_download_family_dataset(self):
        """This should download and check paths of the FamilyDataset"""
        dataset = FamilyDataset()
        self.assertTrue(os.path.exists('family/ontology.owl'))
        self.assertFalse(os.path.exists('family/valid.owl'))
        self.assertFalse(os.path.exists('family/test.owl'))



class TestInstanceOfDataset(TestCase):

    @classmethod
    def tearDownClass(self):
        os.remove('ppi_yeast.tar.gz')
        os.remove('ppi_yeast_slim.tar.gz')
        os.remove('gda_human.tar.gz')
        os.remove('gda_human_el.tar.gz')
        os.remove('gda_mouse.tar.gz')
        os.remove('gda_mouse_el.tar.gz')
        os.remove('family.tar.gz')
        shutil.rmtree('ppi_yeast')
        shutil.rmtree('ppi_yeast_slim')
        shutil.rmtree('gda_human')
        shutil.rmtree('gda_human_el')
        shutil.rmtree('gda_mouse')
        shutil.rmtree('gda_mouse_el')
        shutil.rmtree('family')


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
