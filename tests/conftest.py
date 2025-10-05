import os
import shutil
import logging
import mowl

from mowl.datasets.builtin import FamilyDataset, GDAHumanELDataset, GDAMouseELDataset, \
    PPIYeastSlimDataset, PPIHumanDataset, HPIDataset

logger = logging.getLogger("Downloader")
print("Initializing datasets")


def pytest_configure(config):
    print("Downloading datasets")
    print("Downloading family dataset")
    FamilyDataset()
    print("Downloading gda_human_el dataset")
    GDAHumanELDataset()
    print("Downloading gda_mouse_el dataset")
    GDAMouseELDataset()
    print("Downloading ppi_yeast_slim dataset")
    PPIYeastSlimDataset()
    print("Downloading ppi_human dataset")
    PPIHumanDataset()
    print("Downloading hpi dataset")
    HPIDataset()

# def pytest_unconfigure(config):
    # os.remove('ppi_yeast_slim.tar.gz')
    # os.remove('gda_human_el.tar.gz')
    # os.remove('gda_mouse_el.tar.gz')
    # os.remove('family.tar.gz')
    # shutil.rmtree('ppi_yeast_slim')
    # shutil.rmtree('gda_human_el')
    # shutil.rmtree('gda_mouse_el')
    # shutil.rmtree('family')




