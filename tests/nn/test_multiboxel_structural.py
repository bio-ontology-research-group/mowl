"""Unit tests for MultiBoxEL structural objective (regions, losses, masks, re-seeding)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.models import MultiBoxEL
from mowl.nn import MultiBoxELModule
import mowl.nn.el.multiboxel.losses as L
from mowl.nn.el.structural import (
    DefinitionTable,
    ExistenceTargets,
    Regions,
    WitnessTargets,
    nonempty_measure,
)
from tests.geometric.helpers import _write_t3_exact, fixture_path


def test_class_regions_primitive_padding() -> None:
    """k=2, K=16: slots 0-1 active with raw (cen, |off|), slots 2-15 inactive with cen=0, w=-1."""
    mod = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        objective="structural",
        max_components=16,
    )
    assert mod.max_components == 16
    assert mod.definition_table is None
    assert mod.existence_targets is None

    idx = th.tensor([0, 2], dtype=th.long)
    r = mod.class_regions(idx)
    assert isinstance(r, Regions)
    assert r.cen.shape == (2, 16, 3)
    assert r.w.shape == (2, 16, 3)
    assert r.active.shape == (2, 16)

    assert bool(r.active[:, :2].all().item())
    assert bool((~r.active[:, 2:]).all().item())
    assert bool((r.cen[:, 2:] == 0.0).all().item())
    assert bool((r.w[:, 2:] == -1.0).all().item())

    raw_cen, raw_off = mod.class_boxes(idx)
    assert th.allclose(r.cen[:, :2], raw_cen)
    assert th.allclose(r.w[:, :2], raw_off)

    all_r = mod.all_class_regions()
    assert isinstance(all_r, Regions)
    assert all_r.cen.shape == (4, 16, 3)
    assert all_r.cen.device.type == "cpu"
    assert not all_r.cen.requires_grad
    assert not all_r.w.requires_grad


def test_class_regions_defined_level_one_and_two() -> None:
    """Hand-set raw rows, 2-level definition table: computed slots equal corner arithmetic."""
    # Classes: 0:A, 1:Aux, 2:X1, 3:X2, 4:X3
    # Aux := X1 cap X2 (level 1, 4 slots), A := Aux cap X3 (level 2, 8 slots)
    gci0 = th.tensor(
        [
            [1, 2],  # Aux <= X1
            [1, 3],  # Aux <= X2
            [0, 1],  # A <= Aux
            [0, 4],  # A <= X3
        ],
        dtype=th.long,
    )
    gci1 = th.tensor(
        [
            [2, 3, 1],  # X1 cap X2 <= Aux
            [1, 4, 0],  # Aux cap X3 <= A
        ],
        dtype=th.long,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=th.zeros((0, 2), dtype=th.long),
        gci1=gci1,
        gci1_bot=th.zeros((0, 2), dtype=th.long),
        n_classes=5,
        top_idx=None,
        bottom_idx=None,
    )

    mod = MultiBoxELModule(
        nb_ont_classes=5,
        nb_rels=1,
        embed_dim=1,
        n_components=2,
        objective="structural",
        max_components=16,
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # Put sentinel garbage in defined classes A (0) and Aux (1) to verify they are ignored
        mod.class_center.weight[0] = th.tensor([99.0, 99.0])
        mod.class_offset.weight[0] = th.tensor([99.0, 99.0])
        mod.class_center.weight[1] = th.tensor([-99.0, -99.0])
        mod.class_offset.weight[1] = th.tensor([99.0, 99.0])

        # X1 (2): [0, 4] (c=2, o=2) and [6, 8] (c=7, o=1)
        mod.class_center.weight[2] = th.tensor([2.0, 7.0])
        mod.class_offset.weight[2] = th.tensor([2.0, 1.0])
        # X2 (3): [1, 5] (c=3, o=2) and [10, 12] (c=11, o=1)
        mod.class_center.weight[3] = th.tensor([3.0, 11.0])
        mod.class_offset.weight[3] = th.tensor([2.0, 1.0])
        # X3 (4): [2, 3] (c=2.5, o=0.5) and [0, 10] (c=5, o=5)
        mod.class_center.weight[4] = th.tensor([2.5, 5.0])
        mod.class_offset.weight[4] = th.tensor([0.5, 5.0])

    r = mod.class_regions(th.tensor([1, 0], dtype=th.long))
    # Row 0 is Aux (level 1, 4 active slots):
    # Pairs of X1 ([0,4], [6,8]) and X2 ([1,5], [10,12]):
    # (0,0): [max(0,1), min(4,5)] = [1, 4] -> c=2.5, w=1.5
    # (0,1): [max(0,10), min(4,12)] = [10, 4] -> c=7.0, w=-3.0
    # (1,0): [max(6,1), min(8,5)] = [6, 5] -> c=5.5, w=-0.5
    # (1,1): [max(6,10), min(8,12)] = [10, 8] -> c=9.0, w=-1.0
    assert r.active[0, :4].all()
    assert (~r.active[0, 4:]).all()
    assert r.cen[0, :4, 0].tolist() == pytest.approx([2.5, 7.0, 5.5, 9.0], abs=1e-6)
    assert r.w[0, :4, 0].tolist() == pytest.approx([1.5, -3.0, -0.5, -1.0], abs=1e-6)
    assert (r.w[0, 4:, 0] == -1.0).all()

    # Row 1 is A = Aux cap X3 (level 2, 4 * 2 = 8 active slots):
    assert r.active[1, :8].all()
    assert (~r.active[1, 8:]).all()
    # First two slots of A come from Aux_0 ([1, 4]) intersected with X3_0 ([2, 3]) and X3_1 ([0, 10]):
    # Aux_0 cap X3_0 = [2, 3] -> c=2.5, w=0.5
    # Aux_0 cap X3_1 = [1, 4] -> c=2.5, w=1.5
    assert r.cen[1, 0, 0].item() == pytest.approx(2.5, abs=1e-6)
    assert r.w[1, 0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert r.cen[1, 1, 0].item() == pytest.approx(2.5, abs=1e-6)
    assert r.w[1, 1, 0].item() == pytest.approx(1.5, abs=1e-6)
    assert (r.w[1, 8:, 0] == -1.0).all()


def test_structural_losses_zero_at_exact_t3(tmp_path: Path) -> None:
    """Exact 1-D k=2 T3 configuration at K=4, margin=0, size_reg=0 has 0 loss across all GCIs and extra loss."""
    ds = PathDataset(fixture_path("t3.owl"))
    cm = CanonicalModel.from_dataset(ds)
    model = MultiBoxEL(
        ds,
        embed_dim=1,
        n_components=2,
        margin=0.0,
        objective="structural",
        max_components=4,
        size_reg=0.0,
        canonical_model=cm,
        neg_sampling_gcis=[],
        model_filepath=str(tmp_path / "t3_struct.pt"),
    )
    _write_t3_exact(model)

    # Build tables (as train() does)
    gci0 = model.training_datasets["gci0"][:]
    gci0_bot = (
        model.training_datasets["gci0_bot"][:]
        if len(model.training_datasets["gci0_bot"]) > 0
        else th.zeros((0, 2), dtype=th.long)
    )
    gci1 = model.training_datasets["gci1"][:]
    gci1_bot = model.training_datasets["gci1_bot"][:][:, :2]
    top_idx = model.class_index_dict.get(TOP, None)
    bot_idx = model.class_index_dict.get(BOTTOM, None)
    index_to_iri = {int(idx): iri for iri, idx in model.class_index_dict.items()}
    n_classes = model.module.nb_ont_classes

    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=n_classes,
        k=2,
        max_components=4,
        top_idx=top_idx,
        bottom_idx=bot_idx,
        enabled=True,
    )
    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=gci0_bot,
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=n_classes,
        top_idx=top_idx,
        bottom_idx=bot_idx,
        index_to_iri=index_to_iri,
        cm=cm,
    )
    model.module.set_structural_tables(table, targets, top_idx=top_idx)

    for mode in ("train", "eval"):
        if mode == "train":
            model.module.train()
        else:
            model.module.eval()
        for gci_name, gci_ds in model.training_datasets.items():
            if len(gci_ds) == 0:
                continue
            losses = model.module(gci_ds[:], gci_name)
            assert float(losses.max().item()) == pytest.approx(0.0, abs=1e-12), (
                f"Non-zero {gci_name} loss in {mode} mode: {losses}"
            )
        extra = model.module.structural_extra_loss()
        assert float(extra.item()) == pytest.approx(0.0, abs=1e-12)


def test_soft_aggregation_rejected() -> None:
    """component_aggregation='soft' with objective='structural' or masked gci2_score raises ValueError."""
    with pytest.raises(ValueError, match="component_aggregation"):
        MultiBoxELModule(
            nb_ont_classes=4,
            nb_rels=1,
            embed_dim=2,
            n_components=2,
            objective="structural",
            component_aggregation="soft",
        )
    cen = th.zeros((2, 2, 3))
    off = th.ones((2, 2, 3)) * 0.2
    bump = th.zeros((2, 3))
    cen_r = th.zeros((2, 3))
    off_r = th.ones((2, 3)) * 0.2
    mask = th.ones((2, 2), dtype=th.bool)
    with pytest.raises(ValueError, match="aggregation='hard'"):
        L.gci2_score(
            cen, off, bump, cen, off, bump, cen_r, off_r, cen_r, off_r,
            0.02, 2.0, aggregation="soft", active_c=mask,
        )
    with pytest.raises(ValueError, match="aggregation='hard'"):
        L.gci2_score(
            cen, off, bump, cen, off, bump, cen_r, off_r, cen_r, off_r,
            0.02, 2.0, aggregation="soft", active_d=mask,
        )


def test_role_scores_mask_identity() -> None:
    """active=None equals all-true mask; a masked-out source slot does not change the mean over remaining ones."""
    gen = th.Generator().manual_seed(42)
    b, k, d = 5, 3, 4
    cen_c = th.randn((b, k, d), generator=gen)
    off_c = th.abs(th.randn((b, k, d), generator=gen)) + 0.1
    bump_c = th.randn((b, d), generator=gen)
    cen_d = th.randn((b, k, d), generator=gen)
    off_d = th.abs(th.randn((b, k, d), generator=gen)) + 0.1
    bump_d = th.randn((b, d), generator=gen)
    cen_h = th.randn((b, d), generator=gen)
    off_h = th.abs(th.randn((b, d), generator=gen)) + 0.1
    cen_t = th.randn((b, d), generator=gen)
    off_t = th.abs(th.randn((b, d), generator=gen)) + 0.1
    cen_ind = th.randn((b, d), generator=gen)
    off_ind = th.abs(th.randn((b, d), generator=gen)) + 0.1
    all_true = th.ones((b, k), dtype=th.bool)

    # 1. active=None equals all-true mask across all role/assertion scores
    s2_none = L.gci2_score(
        cen_c, off_c, bump_c, cen_d, off_d, bump_d, cen_h, off_h, cen_t, off_t, 0.02, 2.0
    )
    s2_true = L.gci2_score(
        cen_c,
        off_c,
        bump_c,
        cen_d,
        off_d,
        bump_d,
        cen_h,
        off_h,
        cen_t,
        off_t,
        0.02,
        2.0,
        active_c=all_true,
        active_d=all_true,
    )
    assert th.allclose(s2_none, s2_true, atol=1e-7)

    s3_none = L.gci3_score_boxes(cen_d, off_d, None, cen_h, off_h, bump_c, 0.02)
    s3_true = L.gci3_score_boxes(cen_d, off_d, all_true, cen_h, off_h, bump_c, 0.02)
    assert th.allclose(s3_none, s3_true, atol=1e-7)

    s3b = L.gci3_bot_score_boxes(off_h)
    assert th.allclose(s3b, th.linalg.norm(off_h, dim=-1), atol=1e-7)

    ca_none = L.class_assertion_score_boxes(cen_c, off_c, None, cen_ind, off_ind, 0.02)
    ca_true = L.class_assertion_score_boxes(
        cen_c, off_c, all_true, cen_ind, off_ind, 0.02
    )
    assert th.allclose(ca_none, ca_true, atol=1e-7)

    # 2. A masked-out source slot in C does not change the mean over the remaining active slots
    extra_cen_c = th.full((b, 1, d), 100.0)
    extra_off_c = th.full((b, 1, d), 50.0)
    cen_c_ext = th.cat([cen_c, extra_cen_c], dim=1)
    off_c_ext = th.cat([off_c, extra_off_c], dim=1)
    act_c_ext = th.cat([all_true, th.zeros((b, 1), dtype=th.bool)], dim=1)

    s2_masked_src = L.gci2_score(
        cen_c_ext,
        off_c_ext,
        bump_c,
        cen_d,
        off_d,
        bump_d,
        cen_h,
        off_h,
        cen_t,
        off_t,
        0.02,
        2.0,
        active_c=act_c_ext,
        active_d=all_true,
    )
    assert th.allclose(s2_masked_src, s2_none, atol=1e-7)

    # 3. A masked-out target slot in D does not reduce the min over the remaining active slots
    # Put a giant target box at slot k that would give 0 inclusion if active, but mask it False
    extra_cen_d = th.zeros((b, 1, d))
    extra_off_d = th.full((b, 1, d), 1000.0)
    cen_d_ext = th.cat([cen_d, extra_cen_d], dim=1)
    off_d_ext = th.cat([off_d, extra_off_d], dim=1)
    act_d_ext = th.cat([all_true, th.zeros((b, 1), dtype=th.bool)], dim=1)

    s3_masked_tgt = L.gci3_score_boxes(
        cen_d_ext, off_d_ext, act_d_ext, cen_h, off_h, bump_c, 0.02
    )
    assert th.allclose(s3_masked_tgt, s3_none, atol=1e-7)

    ca_masked_tgt = L.class_assertion_score_boxes(
        cen_d_ext, off_d_ext, act_d_ext, cen_ind, off_ind, 0.02
    )
    ca_orig = L.class_assertion_score_boxes(
        cen_d, off_d, all_true, cen_ind, off_ind, 0.02
    )
    assert th.allclose(ca_masked_tgt, ca_orig, atol=1e-7)


def test_reseed_moves_nested_component() -> None:
    """Class C with component 1 nested in component 0, violated pair (C,D): after reseed, component 1 sits at D's largest slot with half its width, count = 1."""
    mod = MultiBoxELModule(
        nb_ont_classes=2,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        objective="structural",
        margin=0.05,
        max_components=4,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=th.zeros((0, 2), dtype=th.long),
        gci1=th.zeros((0, 3), dtype=th.long),
        n_classes=2,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets(
        classes=th.tensor([0, 1], dtype=th.long),
        pairs=th.tensor([[0, 1]], dtype=th.long),
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # Class 0 (C): comp 0 is [0, 2]^2 (cen=[1, 1], off=[1, 1]);
        # comp 1 is [0.6, 1.4]^2 (cen=[1, 1], off=[0.4, 0.4]) -> nested in comp 0 (dead)
        mod.class_center.weight[0] = th.tensor([1.0, 1.0, 1.0, 1.0])
        mod.class_offset.weight[0] = th.tensor([1.0, 1.0, 0.4, 0.4])

        # Class 1 (D): comp 0 has cen=[10, 12], off=[2.0, 3.0] (ne=4.0, largest);
        # comp 1 has cen=[20, 20], off=[0.5, 0.5] (ne=1.0)
        mod.class_center.weight[1] = th.tensor([10.0, 12.0, 20.0, 20.0])
        mod.class_offset.weight[1] = th.tensor([2.0, 3.0, 0.5, 0.5])

    count = mod.reseed(np.random.default_rng(0))
    assert count == 1
    assert mod.last_reseed_count == 1

    cen_c, off_c = mod.class_boxes(th.tensor([0], dtype=th.long))
    # Component 0 of C is untouched
    assert cen_c[0, 0].tolist() == pytest.approx([1.0, 1.0], abs=1e-6)
    assert off_c[0, 0].tolist() == pytest.approx([1.0, 1.0], abs=1e-6)
    # Component 1 of C moved to D's largest slot (slot 0) with half its width
    assert cen_c[0, 1].tolist() == pytest.approx([10.0, 12.0], abs=1e-6)
    assert off_c[0, 1].tolist() == pytest.approx([1.0, 1.5], abs=1e-6)


def test_reseed_thin_slot_counts_as_dead() -> None:
    """A non-nested slot with min_m |offset| < 1e-3 is classified as dead and relocated."""
    mod = MultiBoxELModule(
        nb_ont_classes=2,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        objective="structural",
        margin=0.05,
        max_components=4,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=th.zeros((0, 2), dtype=th.long),
        gci1=th.zeros((0, 3), dtype=th.long),
        n_classes=2,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets(
        classes=th.zeros(0, dtype=th.long),
        pairs=th.tensor([[0, 1]], dtype=th.long),
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # Class 0 (C): comp 0 at [0, 0] with off=[1.0, 1.0];
        # comp 1 at [-5, -5] (disjoint from comp 0, not nested) with off=[0.5, 5e-4] -> thin (< 1e-3)
        mod.class_center.weight[0] = th.tensor([0.0, 0.0, -5.0, -5.0])
        mod.class_offset.weight[0] = th.tensor([1.0, 1.0, 0.5, 5e-4])

        # Class 1 (D): both components valid and non-nested, far from C
        mod.class_center.weight[1] = th.tensor([10.0, 14.0, 20.0, 20.0])
        mod.class_offset.weight[1] = th.tensor([2.0, 4.0, 1.0, 1.0])

    count = mod.reseed(np.random.default_rng(0))
    assert count == 1
    cen_c, off_c = mod.class_boxes(th.tensor([0], dtype=th.long))
    assert cen_c[0, 0].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert cen_c[0, 1].tolist() == pytest.approx([10.0, 14.0], abs=1e-6)
    assert off_c[0, 1].tolist() == pytest.approx([1.0, 2.0], abs=1e-6)


def test_reseed_skips_partner_with_nonpositive_ne() -> None:
    """Violated pair (C, D) is skipped when partner D's largest active slot has ne <= 0."""
    # Classes: 0:C (primitive), 1:D := X1 cap X2 (defined, disjoint parents -> ne < 0), 2:X1, 3:X2
    gci0 = th.tensor([[1, 2], [1, 3]], dtype=th.long)
    gci1 = th.tensor([[2, 3, 1]], dtype=th.long)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=4,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets(
        classes=th.zeros(0, dtype=th.long),
        pairs=th.tensor([[0, 1]], dtype=th.long),
    )
    mod = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        objective="structural",
        margin=0.05,
        max_components=4,
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # C (0) has comp 1 nested inside comp 0 (dead)
        mod.class_center.weight[0] = th.tensor([0.0, 0.0, 0.0, 0.0])
        mod.class_offset.weight[0] = th.tensor([1.0, 1.0, 0.2, 0.2])
        # X1 (2) and X2 (3) are far apart so D = X1 cap X2 has all pair slots with ne < 0
        mod.class_center.weight[2] = th.tensor([10.0, 10.0, 12.0, 12.0])
        mod.class_offset.weight[2] = th.tensor([0.5, 0.5, 0.5, 0.5])
        mod.class_center.weight[3] = th.tensor([30.0, 30.0, 32.0, 32.0])
        mod.class_offset.weight[3] = th.tensor([0.5, 0.5, 0.5, 0.5])

    count = mod.reseed(np.random.default_rng(0))
    assert count == 0
    _, off_c = mod.class_boxes(th.tensor([0], dtype=th.long))
    assert off_c[0, 1].tolist() == pytest.approx([0.2, 0.2], abs=1e-6)


def test_reseed_uses_defined_partner_materialised_slot() -> None:
    """When partner in a violated pair is a defined class, reseed copies its largest materialised slot, not its raw row."""
    # Classes: 0:C (primitive), 1:A := X1 cap X2 (defined), 2:X1, 3:X2
    gci0 = th.tensor([[1, 2], [1, 3]], dtype=th.long)
    gci1 = th.tensor([[2, 3, 1]], dtype=th.long)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=4,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets(
        classes=th.zeros(0, dtype=th.long),
        pairs=th.tensor([[0, 1]], dtype=th.long),
    )
    mod = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        objective="structural",
        margin=0.05,
        max_components=4,
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # C (0): comp 1 nested in comp 0 around origin
        mod.class_center.weight[0] = th.tensor([0.0, 0.0, 0.0, 0.0])
        mod.class_offset.weight[0] = th.tensor([1.0, 1.0, 0.2, 0.2])
        # Defined class A (1): sentinel garbage in raw embedding rows
        mod.class_center.weight[1] = th.tensor([-99.0, -99.0, -99.0, -99.0])
        mod.class_offset.weight[1] = th.tensor([99.0, 99.0, 99.0, 99.0])
        # X1 (2): comp 0 is [8, 14] x [8, 14] (cen=[11, 11], off=[3, 3]); comp 1 at [50, 50]
        mod.class_center.weight[2] = th.tensor([11.0, 11.0, 50.0, 50.0])
        mod.class_offset.weight[2] = th.tensor([3.0, 3.0, 0.5, 0.5])
        # X2 (3): comp 0 is [10, 16] x [10, 16] (cen=[13, 13], off=[3, 3]); comp 1 at [60, 60]
        # Intersection X1_0 cap X2_0 = [10, 14]^2 -> cen=[12, 12], w=[2, 2] (ne=4.0)
        mod.class_center.weight[3] = th.tensor([13.0, 13.0, 60.0, 60.0])
        mod.class_offset.weight[3] = th.tensor([3.0, 3.0, 0.5, 0.5])

    count = mod.reseed(np.random.default_rng(0))
    assert count == 1
    cen_c, off_c = mod.class_boxes(th.tensor([0], dtype=th.long))
    assert cen_c[0, 1].tolist() == pytest.approx([12.0, 12.0], abs=1e-6)
    assert off_c[0, 1].tolist() == pytest.approx([1.0, 1.0], abs=1e-6)
    # Raw row of defined class A is untouched
    assert mod.class_center.weight[1].tolist() == pytest.approx(
        [-99.0, -99.0, -99.0, -99.0], abs=1e-6
    )


def test_reseed_class_level_inflation_to_margin() -> None:
    """Primitive class target with NE(C) < margin has its largest component clamped to >= margin."""
    mod = MultiBoxELModule(
        nb_ont_classes=2,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        objective="structural",
        margin=0.05,
        max_components=4,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=th.zeros((0, 2), dtype=th.long),
        gci1=th.zeros((0, 3), dtype=th.long),
        n_classes=2,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets(
        classes=th.tensor([0, 1], dtype=th.long),
        pairs=th.zeros((0, 2), dtype=th.long),
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # Class 0: comp 0 has off=[0.005, 0.08] -> ne=0.01 (< 0.05, largest);
        # comp 1 has off=[0.002, 0.002] -> ne=0.004
        mod.class_center.weight[0] = th.tensor([0.0, 0.0, 5.0, 5.0])
        mod.class_offset.weight[0] = th.tensor([0.005, 0.08, 0.002, 0.002])
        # Class 1: already satisfies NE >= 0.05 (off=[0.1, 0.1] -> ne=0.2)
        mod.class_center.weight[1] = th.tensor([0.0, 0.0, 5.0, 5.0])
        mod.class_offset.weight[1] = th.tensor([0.1, 0.1, 0.1, 0.1])

    count = mod.reseed(np.random.default_rng(0))
    assert count == 1
    _, off_c = mod.class_boxes(th.tensor([0], dtype=th.long))
    # Comp 0 clamped to max(|o|, 0.05) = [0.05, 0.08]; comp 1 untouched
    assert off_c[0, 0].tolist() == pytest.approx([0.05, 0.08], abs=1e-6)
    assert off_c[0, 1].tolist() == pytest.approx([0.002, 0.002], abs=1e-6)


def test_reseed_uses_dead_slot_at_most_once_per_call() -> None:
    """A single dead slot on C is used at most once across two violated pairs (C, D1) and (C, D2) in one call."""
    mod = MultiBoxELModule(
        nb_ont_classes=3,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        objective="structural",
        margin=0.05,
        max_components=4,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=th.zeros((0, 2), dtype=th.long),
        gci1=th.zeros((0, 3), dtype=th.long),
        n_classes=3,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets(
        classes=th.zeros(0, dtype=th.long),
        pairs=th.tensor([[0, 1], [0, 2]], dtype=th.long),
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # Class 0 (C): 1 live slot + 1 dead slot (comp 1 nested in comp 0)
        mod.class_center.weight[0] = th.tensor([0.0, 0.0, 0.0, 0.0])
        mod.class_offset.weight[0] = th.tensor([1.0, 1.0, 0.3, 0.3])
        # Class 1 (D1): 2 live non-nested slots, no dead slots
        mod.class_center.weight[1] = th.tensor([10.0, 10.0, 15.0, 15.0])
        mod.class_offset.weight[1] = th.tensor([1.0, 1.0, 1.0, 1.0])
        # Class 2 (D2): 2 live non-nested slots, no dead slots
        mod.class_center.weight[2] = th.tensor([-10.0, -10.0, -15.0, -15.0])
        mod.class_offset.weight[2] = th.tensor([2.0, 2.0, 1.0, 1.0])

    count = mod.reseed(np.random.default_rng(7))
    assert count == 1
    assert mod.last_reseed_count == 1
    cen_c, _ = mod.class_boxes(th.tensor([0], dtype=th.long))
    # Comp 0 of C was live and must remain at [0, 0]
    assert cen_c[0, 0].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    # Comp 1 of C moved to either D1's or D2's largest slot
    assert cen_c[0, 1].tolist() in (
        pytest.approx([10.0, 10.0], abs=1e-6),
        pytest.approx([-10.0, -10.0], abs=1e-6),
    )


def test_existence_targets_with_cm_unknown_iri_uses_asserted_gci0_bot() -> None:
    """When cm is given, unknown IRIs (e.g. ELNormalizer aux classes) fall back to asserted gci0_bot."""
    class _FakeCM:
        concept_names = {"http://mowl.borg/geo#A", "http://mowl.borg/geo#UnsatNamed"}

        def is_satisfiable(self, iri: str) -> bool:
            return iri == "http://mowl.borg/geo#A"

        def joint_satisfiable(self, iris) -> bool:
            return all(i == "http://mowl.borg/geo#A" for i in iris)

    index_to_iri = {
        0: "http://mowl.borg/geo#A",
        1: "http://mowl.borg/geo#UnsatNamed",
        2: "http://mowl.borg/aux/sat_aux",
        3: "http://mowl.borg/aux/unsat_aux",
        4: "http://www.w3.org/2002/07/owl#Nothing",
    }
    gci0_bot = th.tensor([[3, 4]], dtype=th.long)
    gci1 = th.tensor([[0, 2, 0], [0, 3, 0]], dtype=th.long)
    gci1_bot = th.zeros((0, 2), dtype=th.long)

    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=gci0_bot,
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=5,
        top_idx=None,
        bottom_idx=4,
        index_to_iri=index_to_iri,
        cm=_FakeCM(),  # type: ignore[arg-type]
    )
    # 0 (A, sat in cm) and 2 (unknown IRI, not in gci0_bot) included;
    # 1 (UnsatNamed, unsat in cm) and 3 (unknown IRI, in gci0_bot) excluded.
    assert targets.classes.tolist() == [0, 2]
    assert targets.pairs.tolist() == [[0, 2]]


def test_structural_role_and_assertion_dispatch_defined_masks_and_margin() -> None:
    """Module-level structural dispatch for gci2, gci3, class_assertion, object_property_assertion:
    defined classes as source/target, empty-slot masks (including all-empty source in gci2_score_neg),
    and tightening margin (slack < m scores > 0, slack >= m scores 0, margin=0 iff satisfied).
    """
    # Classes: 0:A := X1 cap X2 (defined), 1:X1 (prim), 2:X2 (prim), 3:D (prim), 4:EmptyDef := X1 cap D (defined, empty)
    gci0 = th.tensor([[0, 1], [0, 2], [4, 1], [4, 3]], dtype=th.long)
    gci1 = th.tensor([[1, 2, 0], [1, 3, 4]], dtype=th.long)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    assert table.def_mask.tolist() == [True, False, False, False, True]
    targets = ExistenceTargets(
        classes=th.tensor([0, 1, 2, 3], dtype=th.long),
        pairs=th.zeros((0, 2), dtype=th.long),
    )

    m = 0.2
    mod = MultiBoxELModule(
        nb_ont_classes=5,
        nb_rels=1,
        nb_inds=2,
        embed_dim=1,
        n_components=2,
        margin=m,
        delta=2.0,
        objective="structural",
        max_components=4,
    )
    mod.set_structural_tables(table, targets)

    with th.no_grad():
        # Sentinel raw rows for defined classes 0 and 4 so using raw rows would blow up
        mod.class_center.weight[0] = th.tensor([500.0, 500.0])
        mod.class_offset.weight[0] = th.tensor([50.0, 50.0])
        mod.class_center.weight[4] = th.tensor([-500.0, -500.0])
        mod.class_offset.weight[4] = th.tensor([50.0, 50.0])

        # X1 (1): slot 0 = [-1, 1] (cen=0, off=1), slot 1 = [10, 12] (cen=11, off=1)
        mod.class_center.weight[1] = th.tensor([0.0, 11.0])
        mod.class_offset.weight[1] = th.tensor([1.0, 1.0])
        # X2 (2): slot 0 = [-1, 1] (cen=0, off=1), slot 1 = [20, 22] (cen=21, off=1)
        # => A = X1 cap X2 has 1 non-empty slot [-1, 1] (cen=0, w=1) and 3 empty slots (w < 0)
        mod.class_center.weight[2] = th.tensor([0.0, 21.0])
        mod.class_offset.weight[2] = th.tensor([1.0, 1.0])
        # D (3): slot 0 = [-1.1, 1.1] (cen=0, off=1.1 -> slack 0.1 < m=0.2 over [-1, 1]),
        #        slot 1 = [30, 30] (cen=30, off=0 -> empty, ne=0)
        mod.class_center.weight[3] = th.tensor([0.0, 30.0])
        mod.class_offset.weight[3] = th.tensor([1.1, 0.0])

        # Zero all bumps
        mod.bump_classes.weight.zero_()
        mod.bump_individuals.weight.zero_()

        # Role 0: head = [-1.1, 1.1] (cen=0, off=1.1 -> slack 0.1 < m=0.2 over A=[-1, 1]),
        #         tail = [-1.3, 1.3] (cen=0, off=1.3 -> slack 0.2 >= m=0.2 over D=[-1.1, 1.1])
        mod.head_center.weight[0] = th.tensor([0.0])
        mod.head_offset.weight[0] = th.tensor([1.1])
        mod.tail_center.weight[0] = th.tensor([0.0])
        mod.tail_offset.weight[0] = th.tensor([1.3])

        # Individuals: ind 0 at cen=0, off=0.95 (inside A=[-1, 1] with slack 0.05 < m=0.2)
        #              ind 1 at cen=0, off=0.70 (inside A=[-1, 1] with slack 0.30 >= m=0.2)
        mod.ind_center.weight[0] = th.tensor([0.0])
        mod.ind_offset.weight[0] = th.tensor([0.95])
        mod.ind_center.weight[1] = th.tensor([0.0])
        mod.ind_offset.weight[1] = th.tensor([0.70])

    # 1. Defined class A (0) as gci2 source, primitive D (3) as target:
    # A=[-1, 1] in head=[-1.1, 1.1] has slack 0.1 < m=0.2 -> term_1 = (0.2 - 0.1) = 0.1 > 0;
    # D=[-1.1, 1.1] in tail=[-1.3, 1.3] has slack 0.2 >= m=0.2 -> term_2 = 0.0;
    # gci2_score = (0.1 + 0.0) / 2 = 0.05 > 0.
    gci2_a_d = th.tensor([[0, 0, 3]], dtype=th.long)
    score_slack_lt_m = float(mod.gci2_loss(gci2_a_d).item())
    assert score_slack_lt_m == pytest.approx(0.05, abs=1e-6)

    # At margin = 0, the exact same geometry is satisfied with slack >= 0, so score == 0.0
    mod.margin = 0.0
    assert float(mod.gci2_loss(gci2_a_d).item()) == pytest.approx(0.0, abs=1e-7)
    mod.margin = m

    # When head_offset is widened to 1.25 (slack 0.25 >= m=0.2), gci2_loss becomes 0.0
    with th.no_grad():
        mod.head_offset.weight[0] = th.tensor([1.25])
    assert float(mod.gci2_loss(gci2_a_d).item()) == pytest.approx(0.0, abs=1e-7)

    # 2. Defined class A (0) as gci2 target (D -> r.A):
    # Put tail = [-1.0, 1.0] so A=[-1, 1] in tail has slack 0.0 < m=0.2 (term_2 = 0.2),
    # while D=[-1.1, 1.1] in head=[-1.35, 1.35] has slack 0.25 >= m=0.2 (term_1 = 0.0).
    with th.no_grad():
        mod.head_offset.weight[0] = th.tensor([1.35])
        mod.tail_offset.weight[0] = th.tensor([1.0])
    gci2_d_a = th.tensor([[3, 0, 0]], dtype=th.long)
    assert float(mod.gci2_loss(gci2_d_a).item()) == pytest.approx(0.1, abs=1e-6)

    # 3. Empty-slot mask: set D (3) to [50, 52] (slot 0) and [30, 30] (slot 1, ne=0)
    # so EmptyDef (4) = X1 cap D is disjoint in all slot pairs and has only empty slots.
    with th.no_grad():
        mod.class_center.weight[3] = th.tensor([50.0, 30.0])
        mod.class_offset.weight[3] = th.tensor([1.0, 0.0])
    r_empty = mod.class_regions(th.tensor([4], dtype=th.long))
    assert bool((nonempty_measure(r_empty.w) <= 0.0).all().item())

    # Positive gci2 with all-empty source and target is 0.0;
    # Negative gci2 with all-empty source and target returns 0.0 (and with all-empty source returns only target half)
    gci2_empty_empty = th.tensor([[4, 0, 4]], dtype=th.long)
    assert float(mod.gci2_loss(gci2_empty_empty, neg=False).item()) == 0.0
    assert float(mod.gci2_loss(gci2_empty_empty, neg=True).item()) == 0.0

    # Negative gci2 with all-empty source (4) and non-empty target (0): source half is 0.0
    gci2_empty_a = th.tensor([[4, 0, 0]], dtype=th.long)
    neg_empty_src = float(mod.gci2_loss(gci2_empty_a, neg=True).item())
    # Target A (cen=0, w=1) vs tail (cen=0, w=1) has dist = -2, relu(-2 + 0.2) = 0 -> (delta - 0)^2 = 4.0
    assert neg_empty_src == pytest.approx(4.0, abs=1e-6)

    # 4. gci3 (exists r.C <= A) with defined target A (0):
    # head = [-0.9, 0.9] inside A = [-1, 1] with slack 0.1 < m=0.2 -> score = (0.2 - 0.1)^2 = 0.01 > 0;
    # head = [-0.75, 0.75] inside A = [-1, 1] with slack 0.25 >= m=0.2 -> score = 0.0.
    gci3_row = th.tensor([[0, 1, 0]], dtype=th.long)
    with th.no_grad():
        mod.head_offset.weight[0] = th.tensor([0.9])
    assert float(mod.gci3_loss(gci3_row).item()) == pytest.approx(0.01, abs=1e-6)
    with th.no_grad():
        mod.head_offset.weight[0] = th.tensor([0.75])
    assert float(mod.gci3_loss(gci3_row).item()) == pytest.approx(0.0, abs=1e-7)

    # 5. class_assertion C(a) with defined class A (0):
    # ind 0 (off=0.95) in A (w=1.0) has slack 0.05 < m=0.2 -> score = 0.15 > 0;
    # ind 1 (off=0.70) in A (w=1.0) has slack 0.30 >= m=0.2 -> score = 0.0.
    ca_rows = th.tensor([[0, 0], [1, 0]], dtype=th.long)
    ca_scores = mod.class_assertion_loss(ca_rows)
    assert float(ca_scores[0].item()) == pytest.approx(0.15, abs=1e-6)
    assert float(ca_scores[1].item()) == pytest.approx(0.0, abs=1e-7)

    # 6. object_property_assertion r(a, b):
    # Set head = tail = [-1, 1]; ind 0 (off=0.95, slack 0.05 < m=0.2) vs ind 1 (off=0.70, slack 0.30 >= m=0.2)
    with th.no_grad():
        mod.head_offset.weight[0] = th.tensor([1.0])
        mod.tail_offset.weight[0] = th.tensor([1.0])
    opa_rows = th.tensor([[0, 0, 0], [1, 0, 1]], dtype=th.long)
    opa_scores = mod.object_property_assertion_loss(opa_rows, neg=False)
    assert float(opa_scores[0].item()) == pytest.approx(0.15, abs=1e-6)
    assert float(opa_scores[1].item()) == pytest.approx(0.0, abs=1e-7)

    # 7. At margin = 0, an unsatisfied gci2, gci3, and class_assertion each score > 0:
    # A (0) is [-1, 1] (cen=0, w=1). Set head = [-0.5, 0.5] (smaller than A -> gci2 source A not in head),
    # set head = [-1.5, 1.5] (larger than A -> gci3 head not in A), and ind 0 off = 1.5 (not in A).
    mod.margin = 0.0
    with th.no_grad():
        mod.head_offset.weight[0] = th.tensor([0.5])
    assert float(mod.gci2_loss(gci2_a_d).item()) > 0.0
    with th.no_grad():
        mod.head_offset.weight[0] = th.tensor([1.5])
        mod.ind_offset.weight[0] = th.tensor([1.5])
    assert float(mod.gci3_loss(gci3_row).item()) > 0.0
    assert float(mod.class_assertion_loss(ca_rows[:1]).item()) > 0.0


def test_structural_definitions_rejects_non_bool() -> None:
    """structural_definitions=1 raises ValueError on both MultiBoxEL and PrismEL (and their modules)."""
    from mowl.models import PrismEL
    from mowl.nn import PrismModule

    with pytest.raises(ValueError, match="structural_definitions"):
        MultiBoxELModule(
            nb_ont_classes=4, nb_rels=1, embed_dim=2, structural_definitions=1  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="structural_definitions"):
        PrismModule(
            nb_ont_classes=4, nb_rels=1, embed_dim=4, structural_definitions=1  # type: ignore[arg-type]
        )
    ds = PathDataset(fixture_path("t3.owl"))
    with pytest.raises(ValueError, match="structural_definitions"):
        MultiBoxEL(ds, embed_dim=2, structural_definitions=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="structural_definitions"):
        PrismEL(ds, embed_dim=4, structural_definitions=1)  # type: ignore[arg-type]


def test_bot_agg_module_and_model_sum_vs_mean_and_validation() -> None:
    """bot_agg defaults to 'mean'; 'sum' equals 'mean' * n_active_pairs on a random batch with padding slots; invalid values raise at module and model level."""
    # Default is "mean" on module and model
    mod_default = MultiBoxELModule(
        nb_ont_classes=6,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        objective="structural",
        max_components=8,
    )
    assert mod_default.bot_agg == "mean"

    ds = PathDataset(fixture_path("t3.owl"))
    model_default = MultiBoxEL(ds, embed_dim=2, objective="structural")
    assert model_default.bot_agg == "mean"
    assert model_default.module.bot_agg == "mean"

    model_sum = MultiBoxEL(ds, embed_dim=2, objective="structural", bot_agg="sum")
    assert model_sum.bot_agg == "sum"
    assert model_sum.module.bot_agg == "sum"

    # Invalid values raise at module construction, attribute assignment, and model construction
    for bad in ("max", "avg", "", 1, None):
        with pytest.raises(ValueError, match="bot_agg"):
            MultiBoxELModule(
                nb_ont_classes=4,
                nb_rels=1,
                embed_dim=2,
                objective="structural",
                bot_agg=bad,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="bot_agg"):
            mod_default.bot_agg = bad  # type: ignore[assignment]
        with pytest.raises(ValueError, match="bot_agg"):
            MultiBoxEL(ds, embed_dim=2, objective="structural", bot_agg=bad)  # type: ignore[arg-type]

    # Sum equals mean * n_active_pairs on a random batch with padding slots (k=2, K=8, plus a defined class with 4 active slots)
    # Classes: 0:A := X1 cap X2 (defined, 4 active slots), 1:X1 (prim, 2 active), 2:X2 (prim, 2 active), 3:D (prim, 2 active)
    gci0 = th.tensor([[0, 1], [0, 2]], dtype=th.long)
    gci1 = th.tensor([[1, 2, 0]], dtype=th.long)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=4,
        k=2,
        max_components=8,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets = ExistenceTargets(
        classes=th.tensor([0, 1, 2, 3], dtype=th.long),
        pairs=th.zeros((0, 2), dtype=th.long),
    )

    th.manual_seed(99)
    mod_mean = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        margin=0.05,
        objective="structural",
        max_components=8,
        bot_agg="mean",
    )
    mod_mean.set_structural_tables(table, targets)
    with th.no_grad():
        # Place all primitive components near the origin with large positive widths so all active slots (including A = X1 cap X2) are non-empty and overlap
        mod_mean.class_center.weight.uniform_(-0.1, 0.1)
        mod_mean.class_offset.weight.uniform_(0.5, 1.0)

    mod_s = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        margin=0.05,
        objective="structural",
        max_components=8,
        bot_agg="sum",
    )
    mod_s.load_state_dict(mod_mean.state_dict())
    mod_s.set_structural_tables(table, targets)

    # Batch of gci1_bot rows: (A, D) -> 4 * 2 = 8 active pairs; (X1, D) -> 2 * 2 = 4 active pairs; (A, A) -> 4 * 4 = 16 active pairs
    rows = th.tensor([[0, 3], [1, 3], [0, 0]], dtype=th.long)
    out_mean = mod_mean.gci1_bot_loss(rows)
    out_sum = mod_s.gci1_bot_loss(rows)
    expected_n_pairs = th.tensor([8.0, 4.0, 16.0], dtype=out_mean.dtype)
    assert bool((out_mean > 0.0).all())
    assert th.allclose(out_sum, out_mean * expected_n_pairs, atol=1e-6, rtol=1e-6)


def test_witness_loss_gradients_to_witness_and_primitive_and_defined_classes() -> None:
    """structural_extra_loss with witness_weight > 0 routes gradients to witness_embed, primitive class rows, and primitive definers of defined classes."""
    # Classes: 0:A := X1 cap X2 (defined), 1:X1 (prim), 2:X2 (prim), 3:D (prim)
    gci0 = th.tensor([[0, 1], [0, 2]], dtype=th.long)
    gci1 = th.tensor([[1, 2, 0]], dtype=th.long)
    gci1_bot = th.tensor([[0, 3]], dtype=th.long)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=4,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    assert table.def_mask.tolist() == [True, False, False, False]

    # Build witness targets where w_0 (for A=0) has pos=(0, 0) (defined class A) and neg=(0, 3) (primitive class D)
    wit = WitnessTargets(
        classes=th.tensor([0], dtype=th.long),
        pos=th.tensor([[0, 0]], dtype=th.long),
        neg=th.tensor([[0, 3]], dtype=th.long),
    )
    targets = ExistenceTargets(
        classes=th.zeros(0, dtype=th.long),
        pairs=th.zeros((0, 2), dtype=th.long),
        witness=wit,
    )

    mod = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        margin=0.2,
        objective="structural",
        max_components=4,
        existence_weight=0.0,
        size_reg=0.0,
        witness_weight=1.5,
    )
    assert mod.witness_embed is None
    assert mod.witness_points() is None
    mod.set_structural_tables(table, targets)
    assert mod.witness_embed is not None
    assert mod.witness_embed.weight.shape == (1, 2)

    with th.no_grad():
        # X1 (1): slot 0 = [-1, 1] x [-2, 2] (cen=[0,0], off=[1,2]); slot 1 far away at [20,20]
        mod.class_center.weight[1] = th.tensor([0.0, 0.0, 20.0, 20.0])
        mod.class_offset.weight[1] = th.tensor([1.0, 2.0, 0.5, 0.5])
        # X2 (2): slot 0 = [-2, 2] x [-1, 1] (cen=[0,0], off=[2,1]); slot 1 far away at [30,30]
        # => A = X1 cap X2 has slot 0 = [-1, 1]^2 (limited by X1 in coord 0 and X2 in coord 1)
        mod.class_center.weight[2] = th.tensor([0.0, 0.0, 30.0, 30.0])
        mod.class_offset.weight[2] = th.tensor([2.0, 1.0, 0.5, 0.5])
        # D (3): slot 0 = [1.0, 3.0] x [-1.0, 1.0] (cen=[2.0, 0.0], off=[1.0, 1.0]); slot 1 at [40,40]
        mod.class_center.weight[3] = th.tensor([2.0, 0.0, 40.0, 40.0])
        mod.class_offset.weight[3] = th.tensor([1.0, 1.0, 0.5, 0.5])
        # Put w_0 at [0.95, 0.0]:
        # In A=[-1,1]^2: coord 0 slack is 1.0 - 0.95 = 0.05 < m=0.2 (determined by X1's coord 0)!
        # In D=[1,3]x[-1,1]: coord 0 slack is 1.0 - |0.95 - 2.0| = -0.05 > -m=-0.2 (violates neg margin)!
        mod.witness_embed.weight[0] = th.tensor([0.95, 0.0])

    pts = mod.witness_points()
    assert pts is not None
    assert pts.shape == (1, 2)
    assert not pts.requires_grad
    assert pts[0].tolist() == pytest.approx([0.95, 0.0], abs=1e-6)

    loss = mod.structural_extra_loss()
    # Expected: 1.5 * ((0.2 - 0.05)^2 + (-0.05 + 0.2)^2) = 1.5 * (0.0225 + 0.0225) = 0.0675
    assert loss.item() == pytest.approx(0.0675, abs=1e-6)
    loss.backward()

    assert mod.witness_embed.weight.grad is not None
    assert mod.class_center.weight.grad is not None
    assert mod.class_offset.weight.grad is not None

    # Witness point receives gradient pulling it left (away from D and deeper into A)
    assert mod.witness_embed.weight.grad[0, 0].item() != 0.0
    # Defined class A (row 0) raw embedding row receives exact 0 gradient
    assert bool((mod.class_center.weight.grad[0] == 0.0).all())
    assert bool((mod.class_offset.weight.grad[0] == 0.0).all())
    # Note: for symmetric box [-1, 1] at cen=0, off=1, w - |p - cen| = |off| - (p - cen) = off + cen - p
    # so both class_center[1, 0] and class_offset[1, 0] (X1 slot 0 coord 0) and D's row 3 slot 0 coord 0 get non-zero gradient!
    assert mod.class_center.weight.grad[1, 0].item() != 0.0
    assert mod.class_offset.weight.grad[1, 0].item() != 0.0
    assert mod.class_center.weight.grad[3, 0].item() != 0.0
    assert mod.class_offset.weight.grad[3, 0].item() != 0.0


def test_witness_defaults_leave_structural_extra_loss_unchanged() -> None:
    """With default witness_weight=0.0, witness_embed is not allocated and structural_extra_loss is byte-for-byte unchanged."""
    gci0 = th.tensor([[0, 1], [0, 2]], dtype=th.long)
    gci1 = th.tensor([[1, 2, 0]], dtype=th.long)
    gci1_bot = th.tensor([[0, 3]], dtype=th.long)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=4,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    targets_no_wit = ExistenceTargets.from_normal_forms(
        gci0_bot=th.zeros((0, 2), dtype=th.long),
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=4,
        top_idx=None,
        bottom_idx=None,
        witness=False,
    )
    targets_with_wit = ExistenceTargets.from_normal_forms(
        gci0_bot=th.zeros((0, 2), dtype=th.long),
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=4,
        top_idx=None,
        bottom_idx=None,
        gci0=gci0,
        witness=True,
    )
    assert targets_no_wit.witness is None
    assert targets_with_wit.witness is not None

    th.manual_seed(2026)
    mod_a = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        objective="structural",
        max_components=4,
    )
    mod_a.set_structural_tables(table, targets_no_wit)

    th.manual_seed(2026)
    mod_b = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        objective="structural",
        max_components=4,
    )
    mod_b.set_structural_tables(table, targets_with_wit)

    assert mod_a.witness_weight == 0.0
    assert mod_b.witness_weight == 0.0
    assert mod_a.witness_embed is None
    assert mod_b.witness_embed is None
    assert set(mod_a.state_dict().keys()) == set(mod_b.state_dict().keys())

    extra_a = mod_a.structural_extra_loss()
    extra_b = mod_b.structural_extra_loss()
    assert th.equal(extra_a, extra_b)
    assert th.equal(mod_a.gci1_bot_loss(gci1_bot), mod_b.gci1_bot_loss(gci1_bot))

    # Re-calling set_structural_tables with witness_weight > 0 preserves witness_embed unless W changes
    mod_w = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        objective="structural",
        max_components=4,
        witness_weight=0.5,
    )
    mod_w.set_structural_tables(table, targets_with_wit)
    first_emb = mod_w.witness_embed
    assert first_emb is not None
    mod_w.set_structural_tables(table, targets_with_wit)
    assert mod_w.witness_embed is first_emb

    smaller_wit = WitnessTargets.from_normal_forms(
        classes=th.tensor([0, 1], dtype=th.long),
        gci0=gci0,
        gci1_bot=gci1_bot,
    )
    targets_smaller = ExistenceTargets(
        classes=th.tensor([0, 1], dtype=th.long),
        pairs=targets_with_wit.pairs,
        witness=smaller_wit,
    )
    mod_w.set_structural_tables(table, targets_smaller)
    assert mod_w.witness_embed is not first_emb
    assert mod_w.witness_embed is not None
    assert mod_w.witness_embed.num_embeddings == 2

    # Validation checks on witness_weight
    for bad_w in (-0.1, -1, True, False, "0.5", None):
        with pytest.raises(ValueError, match="witness_weight"):
            MultiBoxELModule(
                nb_ont_classes=4,
                nb_rels=1,
                embed_dim=2,
                objective="structural",
                witness_weight=bad_w,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="witness_weight"):
            mod_w.witness_weight = bad_w  # type: ignore[assignment]



