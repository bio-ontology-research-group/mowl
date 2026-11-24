"""Tests for ExactBoxInit hooks (init_boxes and anchor_weight) on BoxSquaredEL, ELBE, and MultiBoxEL."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric import ExactBoxInit
from mowl.models import BoxSquaredEL, ELBE, MultiBoxEL
from mowl.nn import BoxSquaredELModule, ELBEModule



def test_init_boxes_applied_box2el(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)
    m = BoxSquaredEL(
        t3_dataset,
        embed_dim=7,
        init_boxes=init,
        model_filepath=str(scratch_dir / "b2.pt"),
    )
    c, o, mask = init.as_tensors()
    assert torch.allclose(m.module.class_center.weight[mask], c[mask, 0])
    assert torch.allclose(m.module.class_offset.weight[mask], o[mask, 0])


def test_init_boxes_applied_elbe(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)
    m = ELBE(
        t3_dataset,
        embed_dim=7,
        init_boxes=init,
        model_filepath=str(scratch_dir / "elbe.pt"),
    )
    c, o, mask = init.as_tensors()
    assert torch.allclose(m.module.class_embed.weight[mask], c[mask, 0])
    assert torch.allclose(m.module.class_offset.weight[mask], o[mask, 0])


def test_init_boxes_applied_multiboxel(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, n_components=2)
    m = MultiBoxEL(
        t3_dataset,
        embed_dim=5,
        n_components=2,
        init_boxes=init,
        model_filepath=str(scratch_dir / "mb.pt"),
    )
    c, o, mask = init.as_tensors()
    n_classes = c.shape[0]
    assert torch.allclose(
        m.module.class_center.weight[mask], c.reshape(n_classes, 10)[mask]
    )
    assert torch.allclose(
        m.module.class_offset.weight[mask], o.reshape(n_classes, 10)[mask]
    )


def test_anchor_weight_zero_leaves_regularisation_unchanged(
    t3_dataset: PathDataset, scratch_dir: Path
) -> None:
    init_k1 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)

    b2_zero = BoxSquaredEL(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=0.0,
        model_filepath=str(scratch_dir / "b2_z.pt"),
    )
    with torch.no_grad():
        b2_zero.module.class_center.weight.add_(1.0)
    assert torch.allclose(
        b2_zero.get_regularization_loss(), b2_zero.module.regularization_loss()
    )

    elbe_zero = ELBE(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=0.0,
        model_filepath=str(scratch_dir / "elbe_z.pt"),
    )
    with torch.no_grad():
        elbe_zero.module.class_embed.weight.add_(1.0)
    assert elbe_zero.get_regularization_loss() == 0


@pytest.mark.parametrize("weight", [0.5, 2.0])
def test_anchor_weight_adds_weighted_anchor_loss(
    t3_dataset: PathDataset, scratch_dir: Path, weight: float
) -> None:
    """get_regularization_loss() == module.regularization_loss() + weight * anchor_loss.

    Both terms are recomputed after perturbing the class centres, so the test does
    not assume that the module regulariser ignores ``class_center``.
    """
    init_k1 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)

    b2 = BoxSquaredEL(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=weight,
        model_filepath=str(scratch_dir / "b2_a.pt"),
    )
    with torch.no_grad():
        b2.module.class_center.weight.add_(1.0)
    anchor_b2 = init_k1.anchor_loss(b2.module)
    assert float(anchor_b2.item()) > 0.0
    expected_b2 = b2.module.regularization_loss() + weight * anchor_b2
    assert torch.allclose(b2.get_regularization_loss(), expected_b2, atol=1e-6)

    # ELBEModule defines no regularization_loss, so the base term is 0
    # (EmbeddingELModel.get_regularization_loss falls back to 0).
    assert not hasattr(ELBEModule, "regularization_loss")
    elbe = ELBE(
        t3_dataset,
        embed_dim=7,
        init_boxes=init_k1,
        anchor_weight=weight,
        model_filepath=str(scratch_dir / "elbe_a.pt"),
    )
    with torch.no_grad():
        elbe.module.class_embed.weight.add_(1.0)
    anchor_elbe = init_k1.anchor_loss(elbe.module)
    assert float(anchor_elbe.item()) > 0.0
    expected_elbe = weight * anchor_elbe
    assert torch.allclose(elbe.get_regularization_loss(), expected_elbe, atol=1e-6)

    init_k2 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, n_components=2)
    mb = MultiBoxEL(
        t3_dataset,
        embed_dim=5,
        n_components=2,
        init_boxes=init_k2,
        anchor_weight=weight,
        model_filepath=str(scratch_dir / "mb_a.pt"),
    )
    with torch.no_grad():
        mb.module.class_center.weight.add_(1.0)
    anchor_mb = init_k2.anchor_loss(mb.module)
    assert float(anchor_mb.item()) > 0.0
    expected_mb = mb.module.regularization_loss() + weight * anchor_mb
    assert torch.allclose(mb.get_regularization_loss(), expected_mb, atol=1e-6)


def test_defaults_unchanged(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    torch.manual_seed(12345)
    b2_model = BoxSquaredEL(
        t3_dataset, embed_dim=7, model_filepath=str(scratch_dir / "b2_def.pt")
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
        t3_dataset, embed_dim=7, model_filepath=str(scratch_dir / "elbe_def.pt")
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


def test_negative_anchor_weight_raises(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    with pytest.raises(ValueError, match="anchor_weight"):
        BoxSquaredEL(
            t3_dataset,
            embed_dim=5,
            anchor_weight=-0.1,
            model_filepath=str(scratch_dir / "b2_neg.pt"),
        )
    with pytest.raises(ValueError, match="anchor_weight"):
        ELBE(
            t3_dataset,
            embed_dim=5,
            anchor_weight=-0.1,
            model_filepath=str(scratch_dir / "elbe_neg.pt"),
        )
    with pytest.raises(ValueError, match="anchor_weight"):
        MultiBoxEL(
            t3_dataset,
            embed_dim=5,
            anchor_weight=-0.1,
            model_filepath=str(scratch_dir / "mb_neg.pt"),
        )

