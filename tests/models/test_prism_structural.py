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
from tests.geometric.helpers import NS, _strict_structural_loss, fixture_path


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


def test_prism_structural_survives_reload_and_ablation(
    t3_dataset: PathDataset, scratch_dir: Path
) -> None:
    """Train 3 epochs, save, load_best_model() / from_pretrained(): every GCI loss tensor and n_defined match pre-reload;
    structural_definitions=False disables definition compilation.
    """
    cm = CanonicalModel.from_dataset(t3_dataset)
    ckpt = str(scratch_dir / "prism_reload_t3.pt")

    np.random.seed(0)
    th.manual_seed(0)
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
        model_filepath=ckpt,
    )
    model.train(epochs=3, validate_every=10)
    th.save(model.module.state_dict(), ckpt)

    model.module.eval()
    assert model.module.definition_table is not None
    pre_n_defined = model.module.definition_table.n_defined
    assert pre_n_defined == 3
    with th.no_grad():
        pre_gci = {
            name: model.module(gci_ds[:], name).detach().clone()
            for name, gci_ds in model.training_datasets.items()
            if len(gci_ds) > 0
        }
        pre_extra = model.module.structural_extra_loss().detach().clone()

    model.load_best_model()
    assert model.module.definition_table is not None
    assert model.module.existence_targets is not None
    assert model.module.definition_table.n_defined == pre_n_defined
    with th.no_grad():
        for name, expected in pre_gci.items():
            actual = model.module(model.training_datasets[name][:], name)
            assert th.equal(actual, expected), f"Mismatch on {name} after load_best_model"
        assert th.equal(model.module.structural_extra_loss(), pre_extra)

    fresh = PrismEL(
        t3_dataset,
        embed_dim=4,
        margin=0.02,
        objective="structural",
        max_depth=8,
        existence_weight=1.0,
        size_reg=1e-3,
        canonical_model=cm,
        learning_rate=0.01,
        model_filepath=ckpt,
    )
    fresh.from_pretrained(ckpt)
    fresh.module.eval()
    assert fresh.module.definition_table is not None
    assert fresh.module.definition_table.n_defined == pre_n_defined
    with th.no_grad():
        for name, expected in pre_gci.items():
            actual = fresh.module(model.training_datasets[name][:], name)
            assert th.equal(actual, expected), f"Mismatch on {name} after from_pretrained"
            actual_fresh_rows = fresh.module(fresh.training_datasets[name][:], name)
            assert th.equal(actual_fresh_rows.sort().values, expected.sort().values)
        assert th.equal(fresh.module.structural_extra_loss(), pre_extra)

    # Ablation parity: structural_definitions=False leaves n_defined == 0
    ablated = PrismEL(
        t3_dataset,
        embed_dim=4,
        margin=0.02,
        objective="structural",
        structural_definitions=False,
        canonical_model=cm,
        model_filepath=str(scratch_dir / "prism_ablated_t3.pt"),
    )
    ablated.train(epochs=1, validate_every=10)
    assert ablated.module.definition_table is not None
    assert ablated.module.definition_table.n_defined == 0


@pytest.mark.slow
def test_t3_prism_structural_random_init(t3_dataset: PathDataset, scratch_dir: Path) -> None:
    """5 seeds x 1500 epochs on 8-class T3 fixture, d=3, lr=0.01, margin=0.02, size_reg=1e-2:
    >= 1 seed reaches strict loss <= 1e-6 at margin 0, has positive prism_nonempty_measure
    on all target classes and target pairs, and has an empty triple intersection X1 cap X2 cap X3.
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

        strict_loss = _strict_structural_loss(model)
        mod = model.module
        mod.eval()
        targets = mod.existence_targets
        assert targets is not None
        with th.no_grad():
            cls_regs = mod.class_regions(targets.classes)
            min_cls_ne = float(PL.prism_nonempty_measure(cls_regs).min().item())
            pa_regs = mod.class_regions(targets.pairs[:, 0])
            pb_regs = mod.class_regions(targets.pairs[:, 1])
            pair_regs = PL.prism_regions_intersection(pa_regs, pb_regs)
            min_pair_ne = float(PL.prism_nonempty_measure(pair_regs).min().item())

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

        ok = (
            strict_loss <= 1e-6
            and min_cls_ne > 0.0
            and min_pair_ne > 0.0
            and triple_empty
        )
        if ok:
            successes += 1
        per_seed.append(
            (seed, strict_loss, min_cls_ne, min_pair_ne, triple_ne, triple_empty, ok)
        )

    for seed, strict_loss, min_cls_ne, min_pair_ne, triple_ne, triple_empty, ok in per_seed:
        print(
            f"seed={seed}: strict_loss={strict_loss:.3e}, min_cls_ne={min_cls_ne:.4f}, "
            f"min_pair_ne={min_pair_ne:.4f}, triple_ne={triple_ne:.4f}, "
            f"triple_empty={triple_empty}, ok={ok}"
        )
    assert successes >= 1, f"Expected >= 1 seed to succeed, got {per_seed}"
