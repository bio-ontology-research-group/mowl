"""Benchmark datasets that are downloaded (slow; not run in CI)."""

import pytest

import mowl  # noqa: F401


@pytest.mark.slow
def test_box2el_inference_data():
    from mowl.datasets.builtin.jackermeier2024_reference import \
        Jackermeier2024InferencesDataset
    ds = Jackermeier2024InferencesDataset("GO")
    assert ds.testing.getAxiomCount() >= 2979  # inferences.json has 2,979 pairs
    assert ds.validation.getAxiomCount() >= 330


@pytest.mark.slow
def test_box2el_ppi_data():
    from mowl.datasets.builtin.jackermeier2024_reference import Jackermeier2024PPIDataset
    ds = Jackermeier2024PPIDataset("yeast")
    heads, tails = ds.evaluation_classes
    assert len(heads) == 5502


@pytest.mark.slow
def test_dele_data():
    from mowl.datasets.builtin.mashkova2024 import DELEDataset
    ds = DELEDataset("galen")
    assert ds.testing is not None
    with pytest.raises(ValueError, match="paths"):
        DELEDataset("human_function")


@pytest.mark.slow
def test_emelpp_data():
    from mowl.datasets.builtin.mondal2021 import EmELppSubsumptionDataset
    ds = EmELppSubsumptionDataset("GALEN")
    assert ds.testing.getAxiomCount() > 0


@pytest.mark.slow
def test_cate_ore1_data():
    from mowl.datasets.builtin.zhapa2023 import CatEORE1Dataset
    ds = CatEORE1Dataset()
    assert ds.testing.getAxiomCount() >= 15157
