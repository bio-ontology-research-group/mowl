"""Tests for ExactBoxInit hooks (init_boxes and anchor_weight) on BoxSquaredEL, ELBE, and MultiBoxEL."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric import ExactBoxInit
from mowl.models import BoxSquaredEL, ELBE, MultiBoxEL
from mowl.nn import BoxSquaredELModule, ELBEModule
from tests.geometric.helpers import t3_dataset

__all__ = ["t3_dataset"]

SCRATCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "geometric",
    "_scratch",
)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Return a per-test scratch directory inside the repo's gitignored scratch folder."""
    path = Path(SCRATCH_DIR) / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_init_boxes_applied_box2el(t3_dataset: PathDataset, tmp_path: Path) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)
    m = BoxSquaredEL(
        t3_dataset,
        embed_dim=7,
        init_boxes=init,
        model_filepath=str(tmp_path / "b2.pt"),
    )
    c, o, mask = init.as_tensors()
    assert torch.allclose(m.module.class_center.weight[mask], c[mask, 0])
    assert torch.allclose(m.module.class_offset.weight[mask], o[mask, 0])


def test_init_boxes_applied_elbe(t3_dataset: PathDataset, tmp_path: Path) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)
    m = ELBE(
        t3_dataset,
        embed_dim=7,
        init_boxes=init,
        model_filepath=str(tmp_path / "elbe.pt"),
    )
    c, o, mask = init.as_tensors()
    assert torch.allclose(m.module.class_embed.weight[mask], c[mask, 0])
    assert torch.allclose(m.module.class_offset.weight[mask], o[mask, 0])


def test_init_boxes_applied_multiboxel(t3_dataset: PathDataset, tmp_path: Path) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, n_components=2)
    m = MultiBoxEL(
        t3_dataset,
        embed_dim=5,
        n_components=2,
        init_boxes=init,
        model_filepath=str(tmp_path / "mb.pt"),
    )
    c, o, mask = init.as_tensors()
    n_classes = c.shape[0]
    assert torch.allclose(
        m.module.class_center.weight[mask], c.reshape(n_classes, 10)[mask]
    )
    assert torch.allclose(
        m.module.class_offset.weight[mask], o.reshape(n_classes, 10)[mask]
    )


def test_anchor_weight_adds_to_regularisation(
    t3_dataset: PathDataset, tmp_path: Path
) -> None:
    init_k1 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)

    # BoxSquaredEL: anchor_weight=0 -> equals module.regularization_loss()
    b2_zero = BoxSquaredEL(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=0.0,
        model_filepath=str(tmp_path / "b2_z.pt"),
    )
    with torch.no_grad():
        b2_zero.module.class_center.weight.add_(1.0)
    assert torch.allclose(
        b2_zero.get_regularization_loss(), b2_zero.module.regularization_loss()
    )

    # BoxSquaredEL: anchor_weight=1 -> increases by anchor_loss
    b2_anc = BoxSquaredEL(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=1.0,
        model_filepath=str(tmp_path / "b2_a.pt"),
    )
    base_b2 = b2_anc.module.regularization_loss()
    with torch.no_grad():
        b2_anc.module.class_center.weight.add_(1.0)
    expected_anc_b2 = init_k1.anchor_loss(b2_anc.module)
    assert float(expected_anc_b2.item()) > 0.0
    assert torch.allclose(
        b2_anc.get_regularization_loss() - base_b2, expected_anc_b2, atol=1e-6
    )

    # ELBE: anchor_weight=0 -> 0; anchor_weight=1 -> equals anchor_loss
    elbe_zero = ELBE(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=0.0,
        model_filepath=str(tmp_path / "elbe_z.pt"),
    )
    with torch.no_grad():
        elbe_zero.module.class_embed.weight.add_(1.0)
    assert elbe_zero.get_regularization_loss() == 0

    elbe_anc = ELBE(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=1.0,
        model_filepath=str(tmp_path / "elbe_a.pt"),
    )
    with torch.no_grad():
        elbe_anc.module.class_embed.weight.add_(1.0)
    expected_anc_elbe = init_k1.anchor_loss(elbe_anc.module)
    assert torch.allclose(
        elbe_anc.get_regularization_loss(), expected_anc_elbe, atol=1e-6
    )

    # MultiBoxEL: anchor_weight=1 -> increases by anchor_loss
    init_k2 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, n_components=2)
    mb_anc = MultiBoxEL(
        t3_dataset,
        embed_dim=5,
        n_components=2,
        init_boxes=init_k2,
        anchor_weight=1.0,
        model_filepath=str(tmp_path / "mb_a.pt"),
    )
    base_mb = mb_anc.module.regularization_loss()
    with torch.no_grad():
        mb_anc.module.class_center.weight.add_(1.0)
    expected_anc_mb = init_k2.anchor_loss(mb_anc.module)
    assert torch.allclose(
        mb_anc.get_regularization_loss() - base_mb, expected_anc_mb, atol=1e-6
    )


def test_defaults_unchanged(t3_dataset: PathDataset, tmp_path: Path) -> None:
    torch.manual_seed(12345)
    b2_model = BoxSquaredEL(
        t3_dataset, embed_dim=7, model_filepath=str(tmp_path / "b2_def.pt")
    )
    torch.manual_seed(12345)
    b2_raw = BoxSquaredELModule(
        len(b2_model.class_index_dict),
        len(b2_model.object_property_index_dict),
        len(b2_model.individual_index_dict),
        embed_dim=7,
        gamma=0.02,
        delta=2.5,
        reg_factor=0.2,
    )
    for k, v in b2_model.module.state_dict().items():
        assert torch.equal(v, b2_raw.state_dict()[k]), f"Mismatch in BoxSquaredEL key {k}"

    torch.manual_seed(67890)
    elbe_model = ELBE(
        t3_dataset, embed_dim=7, model_filepath=str(tmp_path / "elbe_def.pt")
    )
    torch.manual_seed(67890)
    elbe_raw = ELBEModule(
        len(elbe_model.class_index_dict),
        len(elbe_model.object_property_index_dict),
        len(elbe_model.individual_index_dict),
        embed_dim=7,
        margin=0,
    )
    for k, v in elbe_model.module.state_dict().items():
        assert torch.equal(v, elbe_raw.state_dict()[k]), f"Mismatch in ELBE key {k}"
