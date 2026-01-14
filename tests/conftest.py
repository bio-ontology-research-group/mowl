import os
import shutil
import logging
import pytest
import mowl

from mowl.datasets.builtin import (
    FamilyDataset as BuiltinFamilyDataset,
    GDAHumanELDataset,
    GDAMouseELDataset,
    PPIYeastSlimDataset as BuiltinPPIYeastSlimDataset,
    PPIHumanDataset,
    HPIDataset,
    GOSubsumptionDataset as BuiltinGOSubsumptionDataset,
    FoodOnSubsumptionDataset as BuiltinFoodOnSubsumptionDataset,
    GDADatasetV2 as BuiltinGDADatasetV2,
)
from tests.datasetFactory import FamilyDataset, PPIYeastSlimDataset, GDADatasetV2

logger = logging.getLogger("Downloader")
print("Initializing datasets")


def pytest_configure(config):
    print("Downloading datasets")
    print("Downloading family dataset")
    BuiltinFamilyDataset()
    print("Downloading gda_human_el dataset")
    GDAHumanELDataset()
    print("Downloading gda_mouse_el dataset")
    GDAMouseELDataset()
    print("Downloading ppi_yeast_slim dataset")
    BuiltinPPIYeastSlimDataset()
    print("Downloading ppi_human dataset")
    PPIHumanDataset()
    print("Downloading hpi dataset")
    HPIDataset()
    print("Downloading go_subsumption dataset")
    BuiltinGOSubsumptionDataset()
    print("Downloading foodon_subsumption dataset")
    BuiltinFoodOnSubsumptionDataset()
    print("Downloading gda_v2 dataset")
    BuiltinGDADatasetV2()


# Session-scoped fixtures for commonly used datasets
# These are loaded once per test session and reused across all tests


@pytest.fixture(scope="session")
def family_dataset():
    """Session-scoped FamilyDataset fixture - loaded once per test session."""
    return FamilyDataset()


@pytest.fixture(scope="session")
def ppi_yeast_slim_dataset():
    """Session-scoped PPIYeastSlimDataset fixture - loaded once per test session."""
    return PPIYeastSlimDataset()


@pytest.fixture(scope="session")
def gda_dataset_v2():
    """Session-scoped GDADatasetV2 fixture - loaded once per test session."""
    return GDADatasetV2()
