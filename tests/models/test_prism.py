"""Tests for the PrismEL model."""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric import ExactBoxInit
from mowl.models import PrismEL
import mowl.nn.el.prism.losses as PL
from tests.datasetFactory import FamilyDataset


def _epoch_loss(model: PrismEL, seed: int) -> float:
    """Compute one training-step total loss (positives + negatives + regularisation) deterministically."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    total = torch.tensor(0.0, device=model.device)
    for gci_name, gci_dataset in model.training_datasets.items():
        if len(gci_dataset) == 0:
            continue
        pos_scores = model.module(gci_dataset[:], gci_name)
        neg_data = model.generate_negatives(gci_name, gci_dataset)
        neg_scores = None
        if neg_data is not None:
            neg_scores = model.module(neg_data, gci_name, neg=True)
        total = total + model.compute_loss(pos_scores, neg_scores)
    total = total + model.get_regularization_loss()
    return float(total.detach().item())


def test_prism_trains_on_family_dataset(scratch_dir: Path) -> None:
    ds = FamilyDataset()
    model = PrismEL(
        ds,
        embed_dim=16,
        margin=0.1,
        model_filepath=str(scratch_dir / "prism_family.pt"),
    )
    losses = []
    model.train(
        epochs=5,
        validate_every=10,
        epoch_callback=lambda ep, m: losses.append(_epoch_loss(m, seed=100 + ep)),
    )
    assert len(losses) == 5
    assert all(np.isfinite(val) for val in losses)
    for param in model.module.parameters():
        assert torch.isfinite(param).all()


def test_prism_count_class_parameters(scratch_dir: Path) -> None:
    ds = FamilyDataset()
    for d in (3, 6, 16, 50):
        model = PrismEL(
            ds,
            embed_dim=d,
            model_filepath=str(scratch_dir / f"prism_d{d}.pt"),
        )
        expected = 2 + 1 + (d - 2) + (d - 2)
        assert expected == 2 * d - 1
        assert model.count_class_parameters() == 2 * d - 1
        actual_sum = (
            model.module.tri_center.weight.shape[1]
            + model.module.tri_inradius.weight.shape[1]
            + model.module.box_center.weight.shape[1]
            + model.module.box_offset.weight.shape[1]
        )
        assert model.count_class_parameters() == actual_sum

    with pytest.raises(ValueError):
        PrismEL(ds, embed_dim=2)
    with pytest.raises(ValueError):
        PrismEL(ds, embed_dim=6, anchor_weight=-0.1)


def test_init_boxes_apply_and_prism_contains_init_box(
    t3_dataset: PathDataset, scratch_dir: Path
) -> None:
    d = 5
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=d, seed=0)
    model = PrismEL(
        t3_dataset,
        embed_dim=d,
        margin=0.0,
        init_boxes=init,
        anchor_weight=0.5,
        model_filepath=str(scratch_dir / "prism_init.pt"),
    )

    c, o, mask = init.as_tensors()
    c_init = c[mask, 0]  # (n_init, d)
    o_init = o[mask, 0]  # (n_init, d)

    # Map init rows to module rows via class_index_dict
    id_to_iri = {idx: iri for iri, idx in init.class_to_id.items()}
    init_rows = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
    mod_rows = torch.tensor(
        [model.class_index_dict[id_to_iri[r]] for r in init_rows], dtype=torch.long
    )

    tc = model.module.tri_center.weight[mod_rows].detach()
    tr = torch.abs(model.module.tri_inradius.weight[mod_rows]).detach()
    bc = model.module.box_center.weight[mod_rows].detach()
    bo = torch.abs(model.module.box_offset.weight[mod_rows]).detach()

    expected_tr = torch.amax(o_init[:, :2], dim=-1, keepdim=True) * math.sqrt(3.0)
    assert torch.allclose(tc, c_init[:, :2], atol=1e-6)
    assert torch.allclose(tr, expected_tr, atol=1e-6)
    assert torch.allclose(bc, c_init[:, 2:], atol=1e-6)
    assert torch.allclose(bo, o_init[:, 2:], atol=1e-6)

    # Verify that the initialised prism CONTAINS the init box:
    # every corner of the init box (coords 0..1 within the triangle via support
    # inequalities; coords 2..d within the box part) satisfies containment (tol 1e-6).
    h = PL.supports(tc, tr)  # (n_init, 3)
    normals = PL.NORMALS.to(dtype=tc.dtype)
    for s0, s1 in itertools.product((-1.0, 1.0), repeat=2):
        corner_2d = c_init[:, :2] + torch.stack(
            [s0 * o_init[:, 0], s1 * o_init[:, 1]], dim=-1
        )
        lhs = corner_2d @ normals.T  # (n_init, 3)
        assert torch.all(lhs <= h + 1e-6)

    lo_box_init = c_init[:, 2:] - o_init[:, 2:]
    hi_box_init = c_init[:, 2:] + o_init[:, 2:]
    lo_prism_box = bc - bo
    hi_prism_box = bc + bo
    assert torch.all(lo_prism_box - 1e-6 <= lo_box_init)
    assert torch.all(hi_box_init <= hi_prism_box + 1e-6)

    # Anchor loss is 0 at init and compares the same quantities
    assert float(init.anchor_loss(model.module).item()) <= 1e-12
    with torch.no_grad():
        model.module.tri_inradius.weight.mul_(-1.0)
        model.module.box_offset.weight.mul_(-1.0)
    assert float(init.anchor_loss(model.module).item()) <= 1e-12

    with torch.no_grad():
        model.module.tri_center.weight.add_(0.5)
    anchor_val = init.anchor_loss(model.module)
    assert float(anchor_val.item()) > 0.0
    expected_reg = model.module.regularization_loss() + 0.5 * anchor_val
    assert torch.allclose(model.get_regularization_loss(), expected_reg, atol=1e-6)
