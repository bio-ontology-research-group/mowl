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
    """component_aggregation='soft' with objective='structural' raises ValueError."""
    with pytest.raises(ValueError, match="component_aggregation"):
        MultiBoxELModule(
            nb_ont_classes=4,
            nb_rels=1,
            embed_dim=2,
            n_components=2,
            objective="structural",
            component_aggregation="soft",
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

    s3b_none = L.gci3_bot_score_boxes(cen_c, off_c, None, off_h)
    s3b_true = L.gci3_bot_score_boxes(cen_c, off_c, all_true, off_h)
    assert th.allclose(s3b_none, s3b_true, atol=1e-7)

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
