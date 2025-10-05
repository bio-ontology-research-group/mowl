import os
import shutil
import logging
import mowl
mowl.init_jvm("10g")

from mowl.datasets.builtin import FamilyDataset, GDAHumanELDataset, GDAMouseELDataset, \
    PPIYeastSlimDataset, GOSubsumptionDataset, FoodOnSubsumptionDataset, GDADatasetV2, PPIHumanDataset, HPIDataset

logger = logging.getLogger("Downloader")

