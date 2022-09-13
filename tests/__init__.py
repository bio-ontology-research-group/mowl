import os
import shutil
import logging
import mowl
mowl.init_jvm("10g")
from mowl.datasets.builtin import FamilyDataset, GDAHumanDataset, GDAHumanELDataset, \
    GDAMouseDataset, GDAMouseELDataset, PPIYeastDataset, PPIYeastSlimDataset


logger = logging.getLogger("Downloader")


def setUpPackage():
    logger.info("Downloading family dataset")
    FamilyDataset()
    logger.info("Downloading gda_human dataset")
    GDAHumanDataset()
    logger.info("Downloading gda_human_el dataset")
    GDAHumanELDataset()
    logger.info("Downloading gda_mouse dataset")
    GDAMouseDataset()
    logger.info("Downloading gda_mouse_el dataset")
    GDAMouseELDataset()
    logger.info("Downloading ppi_yeast dataset")
    PPIYeastDataset()
    logger.info("Downloading ppi_yeast_slim dataset")
    PPIYeastSlimDataset()


def tearDownPackage():
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
