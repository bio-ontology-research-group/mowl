"""Shared fixtures for the model tests that use the ``mowl.geometric`` fixtures."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from mowl.datasets import PathDataset
from tests.geometric.helpers import t3_dataset

__all__ = ["scratch_dir", "t3_dataset"]

SCRATCH_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "geometric" / "_scratch"


@pytest.fixture
def scratch_dir(request: pytest.FixtureRequest, t3_dataset: PathDataset) -> Path:
    """Return a per-test directory under the gitignored ``tests/geometric/_scratch``.

    The directory is named after the test, with characters outside ``[\\w.-]``
    (e.g. parametrisation brackets) replaced by ``_`` so that it is a valid IRI path.

    Also points ``t3_dataset.ontology_path`` into the scratch folder: the models
    write the EL normaliser cache (``<ontology>_mowl_el_normalized_v2.owl``) next
    to ``dataset.ontology_path``, and that cache must not be written into
    ``tests/geometric/fixtures``, which holds only tracked source fixtures.
    """
    path = SCRATCH_DIR / re.sub(r"[^\w.-]", "_", request.node.name)
    path.mkdir(parents=True, exist_ok=True)
    t3_dataset.ontology_path = str(SCRATCH_DIR / "t3.owl")
    return path
