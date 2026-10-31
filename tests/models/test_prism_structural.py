"""Integration tests for PrismEL with objective='structural'."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
from mowl.models import PrismEL
import mowl.nn.el.prism.losses as PL
from tests.geometric.helpers import NS, fixture_path


def test_prism_structural_train_tables(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    """PrismEL(objective='structural') builds DefinitionTable and ExistenceTargets on train()."""
    cm = CanonicalModel.from_dataset(t3_dataset)
    model = PrismEL(
        t3_dataset,
        embed_dim=4,
        margin=0.02,
        objective="structural",
        max_depth=8,
        existence_weight=1.0,
        size_reg=1e-3,
        canonical_model=cm,
        learning_rate=0.01,
        model_filepath=str(scratch_dir / "prism_struct_t3.pt"),
    )
    assert model.module.definition_table is None
    assert model.module.existence_targets is None

    model.train(epochs=2, validate_every=10)

    table = model.module.definition_table
    targets = model.module.existence_targets
    assert table is not None
    assert targets is not None
    assert table.n_defined == 3
    for name in ("A", "B", "C"):
        idx = model.class_index_dict[NS + name]
        assert bool(table.def_mask[idx].item()) is True
    for name in ("X1", "X2", "X3", "Z"):
        idx = model.class_index_dict[NS + name]
        assert bool(table.def_mask[idx].item()) is False

    gci1_rows = model.training_datasets["gci1"][:]
    assert int(table.masked_gci1(gci1_rows).sum().item()) == 3
    assert targets.pairs.shape == (3, 2)
    assert model.additional_training_loss(0) is not None

    # Under structural objective, compute_loss(pos) equals pos.mean() (no double squaring),
    # while the axiom objective retains MSE(pos.mean(), 0) = pos.mean()**2.
    pos = th.tensor([0.2, 0.4, 0.9], dtype=th.float64)
    neg = th.tensor([0.3, 0.7], dtype=th.float64)
    assert th.allclose(model.compute_loss(pos), pos.mean(), atol=1e-12)
    expected_with_neg = pos.mean() + (neg.mean() - 1.0) ** 2
    assert th.allclose(model.compute_loss(pos, neg), expected_with_neg, atol=1e-12)

    axiom_model = PrismEL(
        t3_dataset,
        embed_dim=4,
        margin=0.02,
        objective="axiom",
        model_filepath=str(scratch_dir / "prism_axiom_t3.pt"),
    )
    assert th.allclose(axiom_model.compute_loss(pos), pos.mean() ** 2, atol=1e-12)


@pytest.mark.slow
def test_t3_prism_structural_random_init(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    """5 seeds x 1500 epochs on 8-class T3 fixture, d=3, lr=0.01, margin=0.02, size_reg=1e-2:
    >= 1 seed reaches strict loss <= 1e-6 at margin 0 and has an empty triple intersection X1 cap X2 cap X3.
    """
    cm = CanonicalModel.from_dataset(t3_dataset)

    successes = 0
    per_seed = []

    for seed in range(5):
        np.random.seed(seed)
        th.manual_seed(seed)
        model = PrismEL(
            t3_dataset,
            embed_dim=3,
            margin=0.02,
            objective="structural",
            max_depth=8,
            existence_weight=1.0,
            size_reg=1e-2,
            canonical_model=cm,
            learning_rate=0.01,
            model_filepath=str(scratch_dir / f"prism_t3_seed{seed}.pt"),
        )
        model.train(epochs=1500, validate_every=2000)

        mod = model.module
        mod.eval()
        saved_margin = mod.margin
        saved_size_reg = mod.size_reg
        try:
            mod.margin = 0.0
            mod.size_reg = 0.0
            with th.no_grad():
                strict_loss = 0.0
                for gci_name, gci_ds in model.training_datasets.items():
                    if len(gci_ds) == 0:
                        continue
                    scores = mod(gci_ds[:], gci_name)
                    strict_loss += float(scores.mean().item())
                extra = float(mod.structural_extra_loss().item())
                strict_loss += extra

                x1_idx = model.class_index_dict[NS + "X1"]
                x2_idx = model.class_index_dict[NS + "X2"]
                x3_idx = model.class_index_dict[NS + "X3"]
                regs = mod.class_regions(
                    th.tensor([x1_idx, x2_idx, x3_idx], dtype=th.long, device=model.device)
                )
                r12 = PL.prism_regions_intersection(
                    PL.PrismRegions(h=regs.h[0:1], cen=regs.cen[0:1], w=regs.w[0:1]),
                    PL.PrismRegions(h=regs.h[1:2], cen=regs.cen[1:2], w=regs.w[1:2]),
                )
                r123 = PL.prism_regions_intersection(
                    r12,
                    PL.PrismRegions(h=regs.h[2:3], cen=regs.cen[2:3], w=regs.w[2:3]),
                )
                triple_ne = float(PL.prism_nonempty_measure(r123).item())
                triple_empty = bool(
                    float(r123.h.sum().item()) < 0.0
                    or bool((r123.w < 0.0).any().item())
                )
        finally:
            mod.margin = saved_margin
            mod.size_reg = saved_size_reg

        ok = strict_loss <= 1e-6 and triple_empty
        if ok:
            successes += 1
        per_seed.append((seed, strict_loss, triple_ne, triple_empty, ok))

    for seed, strict_loss, triple_ne, triple_empty, ok in per_seed:
        print(
            f"seed={seed}: strict_loss={strict_loss:.3e}, triple_ne={triple_ne:.4f}, "
            f"triple_empty={triple_empty}, ok={ok}"
        )
    assert successes >= 1, f"Expected >= 1 seed to succeed, got {per_seed}"
