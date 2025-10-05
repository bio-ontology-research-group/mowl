import os
import shutil
import logging
import mowl
mowl.init_jvm("10g")

from mowl.datasets.builtin import FamilyDataset, GDAHumanELDataset, GDAMouseELDataset, \
    PPIYeastSlimDataset, GOSubsumptionDataset, FoodOnSubsumptionDataset, GDADatasetV2, PPIHumanDataset, HPIDataset

logger = logging.getLogger("Downloader")


def setUpModule():
    print("Downloading datasets")
    logger.info("Downloading family dataset")
    FamilyDataset()
    logger.info("Downloading gda_el_human dataset")
    GDAHumanELDataset()
    logger.info("Downloading gda_el_mouse dataset")
    GDAMouseELDataset()
    logger.info("Downloading ppi_yeast_slim dataset")
    PPIYeastSlimDataset()
    logger.info("Downloading go_subsumption dataset")
    GOSubsumptionDataset()
    logger.info("Downloading foodon_subsumption dataset")
    FoodOnSubsumptionDataset()
    logger.info("Downloading gda_v2 dataset")
    GDADatasetV2()
    logger.info("Downloading ppi_human dataset")
    PPIHumanDataset()
    logger.info("Downloading hpi dataset")
    HPIDataset()
