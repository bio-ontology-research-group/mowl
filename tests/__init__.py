import os
import shutil
import logging
import mowl
mowl.init_jvm("5g")
from mowl.datasets.builtin import FamilyDataset, GDAHumanELDataset, GDAMouseELDataset, \
    PPIYeastSlimDataset


logger = logging.getLogger("Downloader")


def setUpPackage():
    logger.info("Downloading family dataset")
    FamilyDataset()
    logger.info("Downloading gda_el_human dataset")
    GDAHumanELDataset()
    logger.info("Downloading gda_el_mouse dataset")
    GDAMouseELDataset()
    logger.info("Downloading ppi_yeast_slim dataset")
    PPIYeastSlimDataset()


# def tearDownPackage():
    # os.remove('ppi_yeast_slim.tar.gz')
    # os.remove('gda_human_el.tar.gz')
    # os.remove('gda_mouse_el.tar.gz')
    # os.remove('family.tar.gz')
    # shutil.rmtree('ppi_yeast_slim')
    # shutil.rmtree('gda_human_el')
    # shutil.rmtree('gda_mouse_el')
    # shutil.rmtree('family')
