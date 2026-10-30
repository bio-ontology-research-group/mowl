"""Tests for the MultiELBE model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric import ExactBoxInit
from mowl.models import ELBE, MultiELBE
from tests.datasetFactory import FamilyDataset


def _epoch_loss(model: ELBE | MultiELBE, seed: int) -> float:
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


def test_model_trains_on_family_dataset(scratch_dir: Path) -> None:
    ds = FamilyDataset()
    model = MultiELBE(
        ds,
        embed_dim=16,
        margin=0.1,
        n_components=2,
        model_filepath=str(scratch_dir / "multielbe_family.pt"),
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


def test_model_k1_matches_elbe_losses_after_init(scratch_dir: Path) -> None:
    ds = FamilyDataset()
    seed = 12345

    np.random.seed(seed)
    torch.manual_seed(seed)
    elbe = ELBE(
        ds,
        embed_dim=16,
        margin=0.1,
        learning_rate=0.001,
        model_filepath=str(scratch_dir / "elbe_k1.pt"),
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    melbe = MultiELBE(
        ds,
        embed_dim=16,
        margin=0.1,
        n_components=1,
        component_reg=0.0,
        learning_rate=0.001,
        model_filepath=str(scratch_dir / "melbe_k1.pt"),
    )

    for k, v in elbe.module.state_dict().items():
        assert torch.equal(v, melbe.module.state_dict()[k]), f"Mismatch in state_dict key {k}"

    loss_elbe_init = _epoch_loss(elbe, seed=999)
    loss_melbe_init = _epoch_loss(melbe, seed=999)
    assert np.isclose(loss_melbe_init, loss_elbe_init, atol=1e-6, rtol=1e-5)

    # Also verify identical loss after 1 training epoch with the same RNG seed
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    elbe.train(epochs=1, validate_every=10)

    np.random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    melbe.train(epochs=1, validate_every=10)

    loss_elbe_after = _epoch_loss(elbe, seed=1000)
    loss_melbe_after = _epoch_loss(melbe, seed=1000)
    assert np.isclose(loss_melbe_after, loss_elbe_after, atol=1e-6, rtol=1e-5)


def test_init_boxes_apply_to_multielbe(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, n_components=2)
    m = MultiELBE(
        t3_dataset,
        embed_dim=5,
        n_components=2,
        init_boxes=init,
        anchor_weight=0.5,
        model_filepath=str(scratch_dir / "melbe_init.pt"),
    )
    c, o, mask = init.as_tensors()
    n_classes = c.shape[0]
    assert torch.allclose(
        m.module.class_embed.weight[mask], c.reshape(n_classes, 10)[mask]
    )
    assert torch.allclose(
        m.module.class_offset.weight[mask], o.reshape(n_classes, 10)[mask]
    )
    assert float(init.anchor_loss(m.module).item()) <= 1e-12

    with torch.no_grad():
        m.module.class_embed.weight.add_(1.0)
    anchor_val = init.anchor_loss(m.module)
    assert float(anchor_val.item()) > 0.0
    expected_reg = m.module.regularization_loss() + 0.5 * anchor_val
    assert torch.allclose(m.get_regularization_loss(), expected_reg, atol=1e-6)
