"""Tests for the MultiBoxEL model and the T3 solvability comparison with BoxSquaredEL."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.models import BoxSquaredEL, MultiBoxEL
from tests.geometric.helpers import NS, t3_dataset

__all__ = ["t3_dataset"]

SCRATCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "geometric",
    "_scratch",
)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest, t3_dataset: PathDataset) -> Path:
    """Return a per-test scratch directory inside the repo's gitignored scratch folder."""
    path = Path(SCRATCH_DIR) / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    t3_dataset.ontology_path = str(Path(SCRATCH_DIR) / "t3.owl")
    return path


def _triple_nonempty(
    boxes_x1: Tuple[np.ndarray, np.ndarray],
    boxes_x2: Tuple[np.ndarray, np.ndarray],
    boxes_x3: Tuple[np.ndarray, np.ndarray],
    tol: float = 1e-6,
) -> bool:
    """Return True iff some component triple (X1_i, X2_j, X3_l) has a non-empty intersection."""
    c1, o1 = boxes_x1
    c2, o2 = boxes_x2
    c3, o3 = boxes_x3
    lo1, hi1 = c1 - np.abs(o1), c1 + np.abs(o1)
    lo2, hi2 = c2 - np.abs(o2), c2 + np.abs(o2)
    lo3, hi3 = c3 - np.abs(o3), c3 + np.abs(o3)
    for i in range(lo1.shape[0]):
        for j in range(lo2.shape[0]):
            for l in range(lo3.shape[0]):
                lo = np.maximum(np.maximum(lo1[i], lo2[j]), lo3[l])
                hi = np.minimum(np.minimum(hi1[i], hi2[j]), hi3[l])
                if bool(np.all(lo <= hi + tol)):
                    return True
    return False


def _is_degenerate(boxes: Tuple[np.ndarray, np.ndarray], eps: float = 1e-3) -> bool:
    """Return True iff every component has min_d(|offset|) <= eps."""
    _, offsets = boxes
    valid_components = np.all(np.abs(offsets) > eps, axis=-1)
    return not bool(np.any(valid_components))


def _total_positive_loss(model: MultiBoxEL | BoxSquaredEL) -> float:
    """Sum over all non-empty training GCI datasets of the mean positive loss."""
    total = 0.0
    with torch.no_grad():
        for gci_name, gci_ds in model.training_datasets.items():
            if len(gci_ds) == 0:
                continue
            scores = model.module(gci_ds[:], gci_name)
            total += float(torch.mean(scores).item())
    return total


def test_multiboxel_trains_on_t3(t3_dataset: PathDataset, tmp_path: Path) -> None:
    m = MultiBoxEL(
        t3_dataset,
        embed_dim=2,
        n_components=2,
        model_filepath=str(tmp_path / "m.pt"),
    )
    m.train(epochs=3, validate_every=10)
    for param in m.module.parameters():
        assert torch.isfinite(param).all()
    loss_val = _total_positive_loss(m)
    assert np.isfinite(loss_val)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "At margin=0.0, Box2EL/MultiBoxEL gci1 (relu(lower-upper)) and gci1_bot "
        "(relu(-dist)) losses vanish on zero-width degenerate boxes touching at a "
        "single boundary point. Measured with seed=0, lr=0.01, neg_sampling_gcis=[]: "
        "(d=2, epochs=1500) MultiBoxEL(k=2) loss=4.365351e-14 (<1e-4), "
        "triple_nonempty=False, deg={'A': False, 'B': False, 'C': True}, "
        "BoxSquaredEL loss=5.937442e-15 (not >1e-3, collapses A,B,C to degenerate "
        "boxes with triple_nonempty=True, deg={'A': True, 'B': True, 'C': True}); "
        "(d=2, epochs=3000) MultiBoxEL(k=2) loss=8.873576e-15, triple_nonempty=False, "
        "deg={'A': False, 'B': False, 'C': True}, BoxSquaredEL loss=4.690546e-16; "
        "(d=3, epochs=1500) MultiBoxEL(k=2) loss=3.962315e-14, triple_nonempty=True, "
        "deg={'A': True, 'B': True, 'C': True}, BoxSquaredEL loss=9.879257e-15; "
        "(d=3, epochs=3000) MultiBoxEL(k=2) loss=8.629650e-14, triple_nonempty=True, "
        "deg={'A': True, 'B': False, 'C': True}, BoxSquaredEL loss=9.259586e-16."
    ),
)
def test_t3_solvability_multibox_vs_box2el(
    t3_dataset: PathDataset, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    mb = MultiBoxEL(
        t3_dataset,
        embed_dim=2,
        n_components=2,
        margin=0.0,
        learning_rate=0.01,
        neg_sampling_gcis=[],
        model_filepath=str(tmp_path / "mb_t3.pt"),
    )
    mb.train(epochs=1500, validate_every=2000)
    mb_loss = _total_positive_loss(mb)

    torch.manual_seed(0)
    b2 = BoxSquaredEL(
        t3_dataset,
        embed_dim=2,
        margin=0.0,
        learning_rate=0.01,
        neg_sampling_gcis=[],
        model_filepath=str(tmp_path / "b2_t3.pt"),
    )
    b2.train(epochs=1500, validate_every=2000)
    b2_loss = _total_positive_loss(b2)

    assert mb_loss < 1e-4, f"Expected MultiBoxEL loss < 1e-4, got {mb_loss}"
    assert b2_loss > 1e-3, f"Expected BoxSquaredEL loss > 1e-3, got {b2_loss}"

    def _get_mb_boxes(iri: str) -> Tuple[np.ndarray, np.ndarray]:
        idx = torch.tensor([mb.class_index_dict[iri]], dtype=torch.long)
        cen, off = mb.module.class_boxes(idx)
        return (
            cen[0].detach().cpu().numpy(),
            off[0].detach().cpu().numpy(),
        )

    assert not _triple_nonempty(
        _get_mb_boxes(NS + "X1"),
        _get_mb_boxes(NS + "X2"),
        _get_mb_boxes(NS + "X3"),
    )
    for name in ("A", "B", "C"):
        assert not _is_degenerate(_get_mb_boxes(NS + name)), f"{name} is degenerate"
