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
                h12, l12, u12 = PL.prism_intersection(
                    regs.h[0:1],
                    (regs.cen[0:1], regs.w[0:1]),
                    regs.h[1:2],
                    (regs.cen[1:2], regs.w[1:2]),
                )
                c12 = (l12 + u12) / 2.0
                w12 = (u12 - l12) / 2.0
                h123, l123, u123 = PL.prism_intersection(
                    h12,
                    (c12, w12),
                    regs.h[2:3],
                    (regs.cen[2:3], regs.w[2:3]),
                )
                c123 = (l123 + u123) / 2.0
                w123 = (u123 - l123) / 2.0
                r123 = PL.PrismRegions(h=h123, cen=c123, w=w123)
                triple_ne = float(PL.prism_nonempty_measure(r123).item())
                triple_empty = bool(
                    float(h123.sum().item()) < 0.0
                    or bool((l123 > u123).any().item())
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
