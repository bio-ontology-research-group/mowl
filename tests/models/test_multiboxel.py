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
    """Return a per-test scratch directory inside the repo's gitignored scratch folder.

    Overrides pytest's built-in ``tmp_path`` (which would live under /tmp) and also
    points ``t3_dataset.ontology_path`` at a path in the scratch folder: the models
    write the EL normaliser cache (``<ontology>_mowl_el_normalized_v2.owl``) next to
    ``dataset.ontology_path``, and that cache must not be written into
    ``tests/geometric/fixtures``, which holds only tracked source fixtures.
    """
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


_W_MIN = 0.05


def _width_floor(module: torch.nn.Module, w_min: float = _W_MIN) -> torch.Tensor:
    """Sum over all class boxes (all components) and all coordinates of relu(w_min - |offset|)."""
    return torch.relu(w_min - torch.abs(module.class_offset.weight)).sum()


class _FlooredMultiBoxEL(MultiBoxEL):
    def get_regularization_loss(self) -> torch.Tensor:
        return super().get_regularization_loss() + _width_floor(self.module)


class _FlooredBoxSquaredEL(BoxSquaredEL):
    def get_regularization_loss(self) -> torch.Tensor:
        return super().get_regularization_loss() + _width_floor(self.module)


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


_OWL_THING = "http://www.w3.org/2002/07/owl#Thing"

# Exact zero-loss 1-D configuration for MultiBoxEL(k=2) on T3, as intervals per component.
_T3_EXACT_1D = {
    "X1": [(0.0, 1.0), (2.0, 3.0)],
    "X2": [(0.0, 1.0), (4.0, 5.0)],
    "X3": [(2.0, 3.0), (4.0, 5.0)],
    "A": [(0.0, 1.0), (0.0, 1.0)],
    "C": [(2.0, 3.0), (2.0, 3.0)],
    "B": [(4.0, 5.0), (4.0, 5.0)],
    "Z": [(0.0, 1.0), (0.0, 1.0)],
}


def _make_t3_multibox(
    dataset: PathDataset, path: Path, learning_rate: float = 0.01
) -> _FlooredMultiBoxEL:
    """Return a floored 1-D MultiBoxEL(k=2) on T3 at margin 0 without negative sampling."""
    return _FlooredMultiBoxEL(
        dataset,
        embed_dim=1,
        n_components=2,
        margin=0.0,
        learning_rate=learning_rate,
        neg_sampling_gcis=[],
        model_filepath=str(path),
    )


def _write_t3_exact(model: MultiBoxEL) -> None:
    """Write the exact 1-D configuration into the class boxes (owl:Thing = [-1, 6] in both components)."""
    intervals = dict(_T3_EXACT_1D)
    with torch.no_grad():
        for name, comps in list(intervals.items()) + [(None, [(-1.0, 6.0), (-1.0, 6.0)])]:
            iri = _OWL_THING if name is None else NS + name
            idx = model.class_index_dict[iri]
            model.module.class_center.weight[idx] = torch.tensor(
                [(lo + hi) / 2 for lo, hi in comps]
            )
            model.module.class_offset.weight[idx] = torch.tensor(
                [(hi - lo) / 2 for lo, hi in comps]
            )


def _mb_boxes(model: MultiBoxEL, name: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = torch.tensor([model.class_index_dict[NS + name]], dtype=torch.long)
    cen, off = model.module.class_boxes(idx)
    return cen[0].detach().cpu().numpy(), off[0].detach().cpu().numpy()


def _assert_t3_multibox_separates(model: MultiBoxEL) -> None:
    for name in ("A", "B", "C"):
        assert not _is_degenerate(_mb_boxes(model, name)), f"{name} is degenerate"
    assert not _triple_nonempty(
        _mb_boxes(model, "X1"), _mb_boxes(model, "X2"), _mb_boxes(model, "X3")
    ), "Expected X1 cap X2 cap X3 to be empty component-wise"


@pytest.mark.slow
def test_t3_expressivity_multibox_zero_loss(
    t3_dataset: PathDataset, tmp_path: Path
) -> None:
    """MultiBoxEL(k=2) in one dimension represents the Helly-violating T3 TBox with zero loss.

    Writes ``X1 = [0,1] u [2,3]``, ``X2 = [0,1] u [4,5]``, ``X3 = [2,3] u [4,5]``,
    ``A = [0,1]``, ``C = [2,3]``, ``B = [4,5]`` (components duplicated), ``Z = [0,1]``,
    ``owl:Thing = [-1,6]`` into the model and checks that at margin 0 every positive
    ``gci0``/``gci1``/``gci1_bot`` row has zero loss, the width floor is zero, and
    ``X1 cap X2 cap X3`` is empty. The test is 1-D because the inherited Box²EL
    disjointness loss demands separation in every coordinate (see the ``gci1_bot``
    docstrings in ``mowl/nn/el/multiboxel/losses.py``); in one dimension
    disjointness and separation coincide.

    Measured: total axiom loss ``0.0``, floor ``0.0``.
    """
    mb = _make_t3_multibox(t3_dataset, tmp_path / "mb_exact.pt")
    _write_t3_exact(mb)

    bad = []
    with torch.no_grad():
        for gci_name in ("gci0", "gci1", "gci1_bot"):
            gci_ds = mb.training_datasets[gci_name]
            rows = gci_ds[:]
            scores = mb.module(rows, gci_name).reshape(-1)
            for row, score in zip(rows.tolist(), scores.tolist()):
                if score > 1e-12:
                    bad.append((gci_name, row, score))
    assert not bad, f"Rows with non-zero loss: {bad}"
    assert _total_positive_loss(mb) <= 1e-12
    assert float(_width_floor(mb.module).item()) == 0.0
    _assert_t3_multibox_separates(mb)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t3_box2el_cannot_reach_zero(
    t3_dataset: PathDataset, tmp_path: Path, seed: int
) -> None:
    """Floored BoxSquaredEL in one dimension cannot reach zero loss on T3.

    The pairwise intersections ``X1 cap X2``, ``X1 cap X3``, ``X2 cap X3`` contain
    ``A``, ``C``, ``B`` of width ``>= 2 * w_min``, so the intervals ``X1, X2, X3``
    intersect pairwise and by Helly's theorem in ``d = 1`` share a common point.
    With ``A == X1 and X2`` and ``B == X2 and X3`` exact, ``A`` and ``B`` then
    overlap, contradicting ``A and B SubClassOf Nothing``; some of the axiom loss or
    the width floor must therefore be positive.

    The argument proves only that the loss is positive; the test asserts
    ``> 1e-4``. Measured (lr 0.01, 3000 epochs, margin 0, w_min 0.05): axiom +
    floor ``4.13e-03`` / ``2.02e-03`` / ``2.02e-03`` for seeds 0 / 1 / 2 (floor 0).
    """
    torch.manual_seed(seed)
    b2 = _FlooredBoxSquaredEL(
        t3_dataset,
        embed_dim=1,
        margin=0.0,
        learning_rate=0.01,
        neg_sampling_gcis=[],
        model_filepath=str(tmp_path / f"b2_t3_{seed}.pt"),
    )
    b2.train(epochs=3000, validate_every=4000)
    b2_loss = _total_positive_loss(b2) + float(_width_floor(b2.module).item())
    assert b2_loss > 1e-4, f"Expected BoxSquaredEL axiom loss + floor > 1e-4, got {b2_loss}"


@pytest.mark.slow
def test_t3_local_learnability_multibox(
    t3_dataset: PathDataset, tmp_path: Path
) -> None:
    """Adam recovers a zero-loss MultiBoxEL(k=2) solution from a perturbed exact configuration.

    Starts from the configuration of :func:`test_t3_expressivity_multibox_zero_loss`
    with ``N(0, 0.1^2)`` noise added to every centre and offset (generator seed 0)
    and trains the floored 1-D model with Adam (lr 0.005, 2000 epochs, margin 0).

    Measured: initial axiom loss ``4.03e-02``, final ``1.48e-16``, floor ``0``
    (noise seeds 1 and 2 also reach ``< 2e-15``; ``sigma = 0.05`` reaches ``0``).
    """
    torch.manual_seed(0)
    mb = _make_t3_multibox(t3_dataset, tmp_path / "mb_local.pt", learning_rate=0.005)
    _write_t3_exact(mb)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for weight in (mb.module.class_center.weight, mb.module.class_offset.weight):
            weight.add_(0.1 * torch.randn(weight.shape, generator=gen))
    mb.train(epochs=2000, validate_every=4000)
    mb_loss = _total_positive_loss(mb)
    assert mb_loss < 1e-4, f"Expected MultiBoxEL axiom loss < 1e-4, got {mb_loss}"
    _assert_t3_multibox_separates(mb)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Optimisation gap: a zero-loss configuration exists "
        "(test_t3_expressivity_multibox_zero_loss) but Adam from random init "
        "plateaus near 1e-3 (best 7.99e-04 over lr {0.005, 0.01, 0.02} x seed "
        "{0, 1, 2} x epochs {3000, 6000}; seed 0, lr 0.01, 3000 epochs: axiom "
        "1.50e-03, X1 cap X2 cap X3 non-empty)."
    ),
)
def test_t3_random_init_multibox_optimisation_gap(
    t3_dataset: PathDataset, tmp_path: Path
) -> None:
    """Adam from random init plateaus near 1e-3 although a zero-loss configuration exists.

    The zero-loss configuration is the one of
    :func:`test_t3_expressivity_multibox_zero_loss`.
    Floored 1-D MultiBoxEL(k=2), margin 0, w_min 0.05, ``neg_sampling_gcis=[]``.
    Sweep of the axiom loss at 3000 / 6000 epochs (floor 0 unless noted)::

        lr     seed 0                      seed 1                      seed 2
        0.01   1.50e-3 / 1.77e-3           1.56e-3 / 1.79e-3           6.00e-3 / 5.92e-3
        0.005  7.99e-4 (fl 1.03e-3)/1.34e-3 1.12e-3 / 1.19e-3          4.19e-3 / 3.48e-3 (fl 3.2e-4)
        0.02   3.41e-3 / 3.42e-3           1.67e-3 / 2.55e-3           5.22e-3 / 6.83e-3

    ``X1 cap X2 cap X3`` is non-empty in every run except lr 0.005 / seed 1 / 3000.
    The test (seed 0, lr 0.01, 3000 epochs) fails first on the empty-triple check.
    """
    torch.manual_seed(0)
    mb = _make_t3_multibox(t3_dataset, tmp_path / "mb_t3.pt")
    mb.train(epochs=3000, validate_every=4000)
    mb_loss = _total_positive_loss(mb)
    mb_min_offset = float(torch.abs(mb.module.class_offset.weight).min().item())
    assert mb_min_offset >= 0.045, (
        f"Expected MultiBoxEL min_d |offset| >= 0.045, got {mb_min_offset}"
    )
    _assert_t3_multibox_separates(mb)
    assert mb_loss < 1e-4, f"Expected MultiBoxEL axiom loss < 1e-4, got {mb_loss}"
