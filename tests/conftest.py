import logging
import pytest

from tests.datasetFactory import (
    FamilyDataset,
    FoodOnSubsumptionDataset,
    GDADatasetV2,
    GDAHumanELDataset,
    GDAMouseELDataset,
    GOSubsumptionDataset,
    HPIDataset,
    PPIHumanDataset,
    PPIYeastSlimDataset,
)

logger = logging.getLogger("Downloader")


# Session-scoped fixtures for commonly used datasets.
# Datasets are downloaded and instantiated lazily on first request rather than
# eagerly in pytest_configure, so test suites that do not use builtin datasets
# (e.g. `pytest tests/nn`) incur zero dataset download or JVM ontology-loading
# overhead at collection/session start.


@pytest.fixture(scope="session")
def family_dataset():
    """Session-scoped FamilyDataset fixture - loaded lazily once per test session."""
    return FamilyDataset()


@pytest.fixture(scope="session")
def ppi_yeast_slim_dataset():
    """Session-scoped PPIYeastSlimDataset fixture - loaded lazily once per test session."""
    return PPIYeastSlimDataset()


@pytest.fixture(scope="session")
def gda_dataset_v2():
    """Session-scoped GDADatasetV2 fixture - loaded lazily once per test session."""
    return GDADatasetV2()


@pytest.fixture(scope="session")
def gda_human_el_dataset():
    """Session-scoped GDAHumanELDataset fixture - loaded lazily once per test session."""
    return GDAHumanELDataset()


@pytest.fixture(scope="session")
def gda_mouse_el_dataset():
    """Session-scoped GDAMouseELDataset fixture - loaded lazily once per test session."""
    return GDAMouseELDataset()


@pytest.fixture(scope="session")
def ppi_human_dataset():
    """Session-scoped PPIHumanDataset fixture - loaded lazily once per test session."""
    return PPIHumanDataset()


@pytest.fixture(scope="session")
def hpi_dataset():
    """Session-scoped HPIDataset fixture - loaded lazily once per test session."""
    return HPIDataset()


@pytest.fixture(scope="session")
def go_subsumption_dataset():
    """Session-scoped GOSubsumptionDataset fixture - loaded lazily once per test session."""
    return GOSubsumptionDataset()


@pytest.fixture(scope="session")
def foodon_subsumption_dataset():
    """Session-scoped FoodOnSubsumptionDataset fixture - loaded lazily once per test session."""
    return FoodOnSubsumptionDataset()


@pytest.fixture
def abox_kb():
    """Build a small EL KB with TBox + ABox covering class and role assertions."""
    from tests.geometric.helpers import build_abox_kb

    return build_abox_kb()


@pytest.fixture
def abox_kb_random():
    """Return 20 seeded random small KBs (seeds 0..19) with 3-6 individuals each."""
    from tests.geometric.helpers import _build_random_abox_kb

    return [(seed, *_build_random_abox_kb(seed)) for seed in range(20)]
