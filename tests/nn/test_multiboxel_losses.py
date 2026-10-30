"""Unit tests for MultiBoxEL losses and MultiBoxELModule."""
from __future__ import annotations

import pytest
import torch

from mowl.nn import BoxSquaredELModule, MultiBoxELModule
import mowl.nn.el.multiboxel.losses as MBL


def make_pair(
    seed: int,
    n_classes: int = 12,
    n_rels: int = 3,
    d: int = 5,
    n_inds: int = 4,
    gamma: float = 0.05,
    delta: float = 2.0,
    reg_factor: float = 0.05,
):
    """Construct a BoxSquaredELModule and MultiBoxELModule(n_components=1) with identical weights."""
    torch.manual_seed(seed)
    b2 = BoxSquaredELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        gamma=gamma,
        delta=delta,
        reg_factor=reg_factor,
    )
    mb = MultiBoxELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=1,
        gamma=gamma,
        delta=delta,
        reg_factor=reg_factor,
        component_reg=0.0,
    )
    with torch.no_grad():
        mb.class_center.weight.copy_(b2.class_center.weight)
        mb.class_offset.weight.copy_(b2.class_offset.weight)
        mb.bump_classes.weight.copy_(b2.bump_classes.weight)
        mb.head_center.weight.copy_(b2.head_center.weight)
        mb.head_offset.weight.copy_(b2.head_offset.weight)
        mb.tail_center.weight.copy_(b2.tail_center.weight)
        mb.tail_offset.weight.copy_(b2.tail_offset.weight)
        mb.ind_center.weight.copy_(b2.ind_center.weight)
        mb.ind_offset.weight.copy_(b2.ind_offset.weight)
        mb.bump_individuals.weight.copy_(b2.bump_individuals.weight)
    return b2, mb


def _random_batch(
    gci: str,
    cols: int,
    batch_size: int,
    n_classes: int,
    n_rels: int,
    n_inds: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Create a random integer batch with valid entity indices for ``gci``."""
    if gci == "gci2":
        c0 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        c1 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        return torch.cat([c0, r, c1], dim=1)
    if gci == "gci3":
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        c0 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        c1 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        return torch.cat([r, c0, c1], dim=1)
    if gci == "gci3_bot":
        return torch.randint(0, n_rels, (batch_size, cols), generator=gen)
    if gci == "class_assertion":
        ind = torch.randint(0, n_inds, (batch_size, 1), generator=gen)
        cls = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        return torch.cat([ind, cls], dim=1)
    if gci == "object_property_assertion":
        i0 = torch.randint(0, n_inds, (batch_size, 1), generator=gen)
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        i1 = torch.randint(0, n_inds, (batch_size, 1), generator=gen)
        return torch.cat([i0, r, i1], dim=1)
    return torch.randint(0, n_classes, (batch_size, cols), generator=gen)


@pytest.mark.parametrize(
    "gci,cols",
    [
        ("gci0", 2),
        ("gci0_bot", 1),
        ("gci1", 3),
        ("gci1_bot", 2),
        ("gci2", 3),
        ("gci3", 3),
        ("gci3_bot", 1),
        ("class_assertion", 2),
        ("object_property_assertion", 3),
    ],
)
def test_k1_identity(gci: str, cols: int) -> None:
    n_classes, n_rels, d, n_inds = 12, 3, 5, 4
    for seed in range(3):
        b2, mb = make_pair(seed, n_classes=n_classes, n_rels=n_rels, d=d, n_inds=n_inds)
        gen = torch.Generator().manual_seed(100 + seed)
        batch = _random_batch(gci, cols, 256, n_classes, n_rels, n_inds, gen)
        out_b2 = b2(batch, gci, neg=False)
        out_mb = mb(batch, gci, neg=False)
        assert out_mb.shape == (256,)
        assert torch.allclose(out_mb, out_b2, atol=1e-6, rtol=1e-5)

        if gci == "object_property_assertion":
            out_b2_neg = b2(batch, gci, neg=True)
            out_mb_neg = mb(batch, gci, neg=True)
            assert torch.allclose(out_mb_neg, out_b2_neg, atol=1e-6, rtol=1e-5)


def test_k1_identity_gci1_with_empty_intersections() -> None:
    n_classes, n_rels, d, n_inds = 12, 3, 5, 4
    for seed in range(3):
        b2, mb = make_pair(seed, n_classes=n_classes, n_rels=n_rels, d=d, n_inds=n_inds)
        with torch.no_grad():
            b2.class_center.weight.mul_(10.0)
            b2.class_offset.weight.fill_(0.01)
            mb.class_center.weight.copy_(b2.class_center.weight)
            mb.class_offset.weight.copy_(b2.class_offset.weight)

        gen = torch.Generator().manual_seed(200 + seed)
        batch = _random_batch("gci1", 3, 256, n_classes, n_rels, n_inds, gen)

        cen_c, off_c = mb.class_boxes(batch[:, 0])
        cen_d, off_d = mb.class_boxes(batch[:, 1])
        lo, up = MBL.multi_intersection(cen_c, off_c, cen_d, off_d)
        empty_rows = (lo[:, 0, 0, :] > up[:, 0, 0, :]).any(dim=-1)
        non_empty_rows = ~empty_rows
        assert int(empty_rows.sum().item()) >= 128
        assert int(non_empty_rows.sum().item()) >= 1

        out_b2 = b2(batch, "gci1", neg=False)
        out_mb = mb(batch, "gci1", neg=False)
        assert torch.allclose(out_mb, out_b2, atol=1e-6, rtol=1e-5)


def test_k1_identity_gci2_negative() -> None:
    n_classes, n_rels, d, n_inds = 12, 3, 5, 4
    for seed in range(3):
        b2, mb = make_pair(seed, n_classes=n_classes, n_rels=n_rels, d=d, n_inds=n_inds)
        gen = torch.Generator().manual_seed(300 + seed)
        batch = _random_batch("gci2", 3, 256, n_classes, n_rels, n_inds, gen)
        out_b2 = b2(batch, "gci2", neg=True)
        out_mb = mb(batch, "gci2", neg=True)
        assert out_mb.shape == (256,)
        assert torch.allclose(out_mb, out_b2, atol=1e-6, rtol=1e-5)


def test_shapes_k3() -> None:
    n_classes, n_rels, d, n_inds, k, B = 10, 3, 6, 4, 3, 17
    mb = MultiBoxELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
    )
    gen = torch.Generator().manual_seed(42)
    idx = torch.randint(0, n_classes, (B,), generator=gen)
    cen, off = mb.class_boxes(idx)
    assert cen.shape == (B, k, d)
    assert off.shape == (B, k, d)

    lo, up = MBL.multi_intersection(cen, off, cen, off)
    assert lo.shape == (B, k, k, d)
    assert up.shape == (B, k, k, d)
    inc = MBL.multi_inclusion(cen, off, cen, off, gamma=0.0)
    assert inc.shape == (B, k, k)

    for gci, cols in [
        ("gci0", 2),
        ("gci0_bot", 1),
        ("gci1", 3),
        ("gci1_bot", 2),
        ("gci2", 3),
        ("gci3", 3),
        ("gci3_bot", 1),
        ("class_assertion", 2),
        ("object_property_assertion", 3),
    ]:
        batch = _random_batch(gci, cols, B, n_classes, n_rels, n_inds, gen)
        out = mb(batch, gci, neg=False)
        assert out.shape == (B,)
        if gci in mb.neg_capable_gcis:
            out_neg = mb(batch, gci, neg=True)
            assert out_neg.shape == (B,)


def test_gci0_zero_iff_each_component_inside_some_component() -> None:
    mb = MultiBoxELModule(
        nb_ont_classes=3,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        gamma=0.0,
    )
    with torch.no_grad():
        # C (row 0): comp 0 = [0, 2]^2 (c=(1,1), o=(1,1)), comp 1 = [6, 8]^2 (c=(7,7), o=(1,1))
        mb.class_center.weight[0] = torch.tensor([1.0, 1.0, 7.0, 7.0])
        mb.class_offset.weight[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # D_good (row 1): comp 0 = [-1, 3]^2 (contains C_0), comp 1 = [5, 9]^2 (contains C_1)
        mb.class_center.weight[1] = torch.tensor([1.0, 1.0, 7.0, 7.0])
        mb.class_offset.weight[1] = torch.tensor([2.0, 2.0, 2.0, 2.0])

        # D_bad (row 2): comp 0 = [-1, 3]^2 (contains C_0), comp 1 = [20, 22]^2 (misses C_1)
        mb.class_center.weight[2] = torch.tensor([1.0, 1.0, 21.0, 21.0])
        mb.class_offset.weight[2] = torch.tensor([2.0, 2.0, 1.0, 1.0])

    loss_good = mb(torch.tensor([[0, 1]]), "gci0")
    loss_bad = mb(torch.tensor([[0, 2]]), "gci0")
    assert float(loss_good.item()) <= 1e-12
    assert float(loss_bad.item()) > 1.0


def test_gci1_empty_pairs_impose_nothing() -> None:
    mb = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        gamma=0.0,
    )
    with torch.no_grad():
        # C (row 0): C_0 = [0, 4]^2 (c=(2,2), o=(2,2)), C_1 = [100, 102]^2 (c=(101,101), o=(1,1))
        mb.class_center.weight[0] = torch.tensor([2.0, 2.0, 101.0, 101.0])
        mb.class_offset.weight[0] = torch.tensor([2.0, 2.0, 1.0, 1.0])

        # D (row 1): 1 effective component duplicated: D_0 = D_1 = [2, 6]^2 (c=(4,4), o=(2,2))
        # Intersection C_0 cap D = [2, 4]^2; C_1 cap D = empty.
        mb.class_center.weight[1] = torch.tensor([4.0, 4.0, 4.0, 4.0])
        mb.class_offset.weight[1] = torch.tensor([2.0, 2.0, 2.0, 2.0])

        # E_good (row 2): E_0 = E_1 = [1, 5]^2 (c=(3,3), o=(2,2)) -> contains [2, 4]^2
        mb.class_center.weight[2] = torch.tensor([3.0, 3.0, 3.0, 3.0])
        mb.class_offset.weight[2] = torch.tensor([2.0, 2.0, 2.0, 2.0])

        # E_bad (row 3): E_0 = E_1 = [50, 52]^2 -> does not contain [2, 4]^2
        mb.class_center.weight[3] = torch.tensor([51.0, 51.0, 51.0, 51.0])
        mb.class_offset.weight[3] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    loss_good = mb(torch.tensor([[0, 1, 2]]), "gci1")
    loss_bad = mb(torch.tensor([[0, 1, 3]]), "gci1")
    assert float(loss_good.item()) <= 1e-12
    assert float(loss_bad.item()) > 1.0


def test_gci1_bot_requires_all_pairs_disjoint() -> None:
    mb = MultiBoxELModule(
        nb_ont_classes=3,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        gamma=0.0,
    )
    with torch.no_grad():
        # C (row 0): C_0 = [0, 2]^2 (c=(1,1), o=(1,1)), C_1 = [10, 12]^2 (c=(11,11), o=(1,1))
        mb.class_center.weight[0] = torch.tensor([1.0, 1.0, 11.0, 11.0])
        mb.class_offset.weight[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # D_disjoint (row 1): D_0 = [4, 6]^2, D_1 = [20, 22]^2 -> all 4 pairs disjoint
        mb.class_center.weight[1] = torch.tensor([5.0, 5.0, 21.0, 21.0])
        mb.class_offset.weight[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # D_overlap (row 2): D_0 = [4, 6]^2, D_1 = [10.5, 12.5]^2 -> (C_1, D_1) overlaps
        mb.class_center.weight[2] = torch.tensor([5.0, 5.0, 11.5, 11.5])
        mb.class_offset.weight[2] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    loss_disjoint = mb(torch.tensor([[0, 1]]), "gci1_bot")
    loss_overlap = mb(torch.tensor([[0, 2]]), "gci1_bot")
    assert float(loss_disjoint.item()) <= 1e-12
    assert float(loss_overlap.item()) > 0.1


def test_effective_components_counts_nested_and_degenerate() -> None:
    mb = MultiBoxELModule(
        nb_ont_classes=5,
        nb_rels=1,
        embed_dim=2,
        n_components=3,
    )
    with torch.no_grad():
        # Row 0: 3 disjoint non-degenerate boxes -> 3
        mb.class_center.weight[0] = torch.tensor([0.0, 0.0, 5.0, 5.0, 10.0, 10.0])
        mb.class_offset.weight[0] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Row 1: comp 0 ([0, 2]^2) nested inside comp 1 ([-1, 3]^2), comp 2 ([10, 12]^2) separate -> 2
        mb.class_center.weight[1] = torch.tensor([1.0, 1.0, 1.0, 1.0, 11.0, 11.0])
        mb.class_offset.weight[1] = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])

        # Row 2: comp 0 non-degenerate, comp 1 degenerate in one coord (0.0005 <= 1e-3), comp 2 nested in comp 0 -> 1
        mb.class_center.weight[2] = torch.tensor([0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
        mb.class_offset.weight[2] = torch.tensor([2.0, 2.0, 1.0, 0.0005, 1.0, 1.0])

        # Row 3: all 3 components identical non-degenerate boxes -> 1
        mb.class_center.weight[3] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        mb.class_offset.weight[3] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Row 4: all 3 components degenerate -> 0
        mb.class_center.weight[4] = torch.zeros(6)
        mb.class_offset.weight[4] = torch.zeros(6)

    eff = mb.effective_components(eps=1e-3)
    assert eff.shape == (5,)
    assert eff.tolist() == [3, 2, 1, 1, 0]


def test_regularization_matches_box2el_when_component_reg_zero() -> None:
    b2, mb = make_pair(seed=7, reg_factor=0.15)
    assert torch.allclose(mb.regularization_loss(), b2.regularization_loss(), atol=1e-6)

    # With k=2 and component_reg > 0, collapsing component centres increases regularization_loss
    mb_k2_zero = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=2,
        embed_dim=3,
        n_components=2,
        reg_factor=0.1,
        component_reg=0.0,
    )
    mb_k2_reg = MultiBoxELModule(
        nb_ont_classes=4,
        nb_rels=2,
        embed_dim=3,
        n_components=2,
        reg_factor=0.1,
        component_reg=0.5,
    )
    with torch.no_grad():
        # Set identical centres across components so ||c_{C,0} - c_{C,1}||_2 == 0 < tau (0.1)
        mb_k2_zero.class_center.weight.zero_()
        mb_k2_reg.class_center.weight.zero_()
        mb_k2_reg.bump_classes.weight.copy_(mb_k2_zero.bump_classes.weight)

    base_reg = mb_k2_zero.regularization_loss()
    penalised_reg = mb_k2_reg.regularization_loss()
    expected_extra = 0.5 * 0.1
    assert torch.allclose(penalised_reg - base_reg, torch.tensor(expected_extra), atol=1e-6)


def test_exact_box_init_and_gradients_with_multiboxel() -> None:
    import numpy as np
    from mowl.geometric.init import ExactBoxInit

    n_classes, k, d = 6, 2, 4
    centers = np.ones((n_classes, k, d), dtype=np.float32) * 0.5
    offsets = np.ones((n_classes, k, d), dtype=np.float32) * 0.25
    centers[-1, :, :] = np.nan
    offsets[-1, :, :] = np.nan
    init = ExactBoxInit(
        centers=centers,
        offsets=offsets,
        class_to_id={f"http://ex.org/C{i}": i for i in range(n_classes)},
        method="projected",
        n_components=k,
        seed=0,
        report={},
    )
    mb = MultiBoxELModule(nb_ont_classes=n_classes, nb_rels=2, embed_dim=d, n_components=k)
    written = init.apply_to_module(mb)
    assert written == n_classes - 1
    assert float(init.anchor_loss(mb).item()) <= 1e-12

    batch = torch.tensor([[0, 1, 2], [2, 3, 4]])
    loss = mb(batch, "gci1").sum() + init.anchor_loss(mb)
    loss.backward()
    assert mb.class_center.weight.grad is not None
    assert torch.isfinite(mb.class_center.weight.grad).all()
    assert mb.class_offset.weight.grad is not None
    assert torch.isfinite(mb.class_offset.weight.grad).all()

    with pytest.raises(ValueError):
        mb(torch.tensor([[0, 1]]), "class_assertion")
    with pytest.raises(ValueError):
        mb(torch.tensor([[0, 0, 1]]), "object_property_assertion")


def test_resolve_n_components_rejects_ambiguous_embedding() -> None:
    bare_emb = torch.nn.Embedding(6, 8)
    with pytest.raises(ValueError):
        MBL._resolve_n_components(bare_emb)
    with pytest.raises(ValueError):
        MBL._resolve_n_components(bare_emb, embed_dim=4)
    assert MBL._resolve_n_components(bare_emb, embed_dim=8) == 1
    assert MBL._resolve_n_components(bare_emb, n_components=2, embed_dim=4) == 2
    bare_emb.embed_dim = 8
    assert MBL._resolve_n_components(bare_emb) == 1


def test_multiboxel_soft_aggregation_and_diversity_reg() -> None:
    n_classes, n_rels, d, n_inds, k = 6, 2, 4, 3, 3
    torch.manual_seed(33)
    mb_hard = MultiBoxELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
        gamma=0.05,
        component_aggregation="hard",
        diversity_reg=0.0,
    )
    torch.manual_seed(33)
    mb_soft = MultiBoxELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
        gamma=0.05,
        component_aggregation="soft",
        tau=1e-4,
        diversity_reg=0.2,
    )
    gen = torch.Generator().manual_seed(77)
    batch = _random_batch("gci1", 3, 16, n_classes, n_rels, n_inds, gen)
    assert torch.allclose(mb_soft(batch, "gci1"), mb_hard(batch, "gci1"), atol=1e-3, rtol=1e-3)

    with torch.no_grad():
        mb_soft.class_center.weight.zero_()
        mb_soft.class_offset.weight.fill_(1.0)
        mb_hard.class_center.weight.zero_()
        mb_hard.class_offset.weight.fill_(1.0)
    assert float(mb_soft.regularization_loss().item()) > float(mb_hard.regularization_loss().item())

    with pytest.raises(ValueError, match="component_aggregation"):
        MultiBoxELModule(nb_ont_classes=4, nb_rels=2, component_aggregation="invalid")



