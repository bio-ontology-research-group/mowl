"""Unit tests for MultiELBE losses and MultiELBEModule."""
from __future__ import annotations

import pytest
import torch

from mowl.nn import ELBEModule, MultiELBEModule
import mowl.nn.el.elbe.losses as EL
import mowl.nn.el.multielbe.losses as MEL


def make_pair(
    seed: int,
    n_classes: int = 12,
    n_rels: int = 3,
    d: int = 5,
    n_inds: int = 4,
    margin: float = 0.1,
):
    """Construct an ELBEModule and MultiELBEModule(n_components=1) with identical weights."""
    torch.manual_seed(seed)
    elbe = ELBEModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        margin=margin,
    )
    me = MultiELBEModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        margin=margin,
        n_components=1,
        component_reg=0.0,
    )
    with torch.no_grad():
        me.class_embed.weight.copy_(elbe.class_embed.weight)
        me.class_offset.weight.copy_(elbe.class_offset.weight)
        me.rel_embed.weight.copy_(elbe.rel_embed.weight)
        me.ind_embed.weight.copy_(elbe.ind_embed.weight)
        me.ind_offset.weight.copy_(elbe.ind_offset.weight)
    return elbe, me


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
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        c0 = torch.randint(0, n_classes, (batch_size, max(1, cols)), generator=gen)
        return torch.cat([r, c0], dim=1)
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
def test_k1_identity_with_elbe(gci: str, cols: int) -> None:
    n_classes, n_rels, d, n_inds = 12, 3, 5, 4
    for seed in range(3):
        elbe, me = make_pair(
            seed, n_classes=n_classes, n_rels=n_rels, d=d, n_inds=n_inds
        )
        gen = torch.Generator().manual_seed(100 + seed)
        batch = _random_batch(gci, cols, 256, n_classes, n_rels, n_inds, gen)
        out_elbe = elbe(batch, gci, neg=False)
        out_me = me(batch, gci, neg=False)
        assert out_me.shape == out_elbe.shape
        assert out_me.shape[0] == 256
        assert torch.allclose(out_me, out_elbe, atol=1e-6, rtol=1e-5)

        if gci in ("gci2", "object_property_assertion"):
            out_elbe_neg = elbe(batch, gci, neg=True)
            out_me_neg = me(batch, gci, neg=True)
            assert out_me_neg.shape == out_elbe_neg.shape
            assert torch.allclose(out_me_neg, out_elbe_neg, atol=1e-6, rtol=1e-5)


def test_k1_identity_gci1_with_empty_intersections() -> None:
    n_classes, n_rels, d, n_inds = 12, 3, 5, 4
    for seed in range(3):
        elbe, me = make_pair(
            seed, n_classes=n_classes, n_rels=n_rels, d=d, n_inds=n_inds
        )
        with torch.no_grad():
            elbe.class_embed.weight.mul_(10.0)
            elbe.class_offset.weight.fill_(0.01)
            me.class_embed.weight.copy_(elbe.class_embed.weight)
            me.class_offset.weight.copy_(elbe.class_offset.weight)

        gen = torch.Generator().manual_seed(200 + seed)
        batch = _random_batch("gci1", 3, 256, n_classes, n_rels, n_inds, gen)

        cen_c, off_c = me.class_boxes(batch[:, 0])
        cen_d, off_d = me.class_boxes(batch[:, 1])
        lo, up = MEL.multi_intersection(cen_c, off_c, cen_d, off_d)
        empty_rows = (lo[:, 0, 0, :] > up[:, 0, 0, :]).any(dim=-1)
        non_empty_rows = ~empty_rows
        assert int(empty_rows.sum().item()) >= 128
        assert int(non_empty_rows.sum().item()) >= 1

        out_elbe = elbe(batch, "gci1", neg=False)
        out_me = me(batch, "gci1", neg=False)
        assert out_me.shape == out_elbe.shape
        assert torch.allclose(out_me, out_elbe, atol=1e-6, rtol=1e-5)


def test_shapes_k3() -> None:
    n_classes, n_rels, d, n_inds, k, B = 10, 3, 6, 4, 3, 17
    me = MultiELBEModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
    )
    gen = torch.Generator().manual_seed(42)
    idx = torch.randint(0, n_classes, (B,), generator=gen)
    cen, off = me.class_boxes(idx)
    assert cen.shape == (B, k, d)
    assert off.shape == (B, k, d)

    lo, up = MEL.multi_intersection(cen, off, cen, off)
    assert lo.shape == (B, k, k, d)
    assert up.shape == (B, k, k, d)
    inc = MEL.multi_inclusion(cen, off, cen, off, margin=0.0)
    assert inc.shape == (B, k, k)
    pair_sc = MEL.multi_pair_score(EL.box_inclusion_score, cen, off, cen, off, 0.0)
    assert pair_sc.shape == (B, k, k)
    assert torch.allclose(inc, pair_sc, atol=1e-6)

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
        out = me(batch, gci, neg=False)
        assert out.shape[0] == B
        assert out.numel() == B
        if gci in me.neg_capable_gcis:
            out_neg = me(batch, gci, neg=True)
            assert out_neg.shape[0] == B
            assert out_neg.numel() == B


def test_gci0_zero_iff_each_component_inside_some_component() -> None:
    me = MultiELBEModule(
        nb_ont_classes=3,
        nb_rels=1,
        embed_dim=2,
        n_components=2,
        margin=0.0,
    )
    with torch.no_grad():
        # C (row 0): comp 0 = [0, 2]^2 (c=(1,1), o=(1,1)), comp 1 = [6, 8]^2 (c=(7,7), o=(1,1))
        me.class_embed.weight[0] = torch.tensor([1.0, 1.0, 7.0, 7.0])
        me.class_offset.weight[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # D_good (row 1): comp 0 = [-1, 3]^2 (contains C_0), comp 1 = [5, 9]^2 (contains C_1)
        me.class_embed.weight[1] = torch.tensor([1.0, 1.0, 7.0, 7.0])
        me.class_offset.weight[1] = torch.tensor([2.0, 2.0, 2.0, 2.0])

        # D_bad (row 2): comp 0 = [-1, 3]^2 (contains C_0), comp 1 = [20, 22]^2 (misses C_1)
        me.class_embed.weight[2] = torch.tensor([1.0, 1.0, 21.0, 21.0])
        me.class_offset.weight[2] = torch.tensor([2.0, 2.0, 1.0, 1.0])

    loss_good = me(torch.tensor([[0, 1]]), "gci0")
    loss_bad = me(torch.tensor([[0, 2]]), "gci0")
    assert float(loss_good.item()) <= 1e-12
    assert float(loss_bad.item()) > 1.0


def test_gci3_keeps_elbe_overlap_semantics() -> None:
    me = MultiELBEModule(
        nb_ont_classes=3,
        nb_rels=1,
        embed_dim=2,
        n_components=1,
        margin=0.0,
    )
    with torch.no_grad():
        # r (row 0): zero translation (0, 0)
        me.rel_embed.weight[0] = torch.tensor([0.0, 0.0])

        # C (row 0): [0, 4]^2 -> c=(2,2), off_c=(2,2). Translated box C - r = [0, 4]^2.
        me.class_embed.weight[0] = torch.tensor([2.0, 2.0])
        me.class_offset.weight[0] = torch.tensor([2.0, 2.0])

        # D_overlap (row 1): [3, 5]^2 -> d=(4,4), off_d=(1,1).
        # Overlaps C - r on [3, 4]^2, but C - r is NOT included in D_overlap (off_c=2 > off_d=1).
        me.class_embed.weight[1] = torch.tensor([4.0, 4.0])
        me.class_offset.weight[1] = torch.tensor([1.0, 1.0])

        # D_disjoint (row 2): [10, 12]^2 -> d=(11,11), off_d=(1,1). Disjoint from C - r.
        me.class_embed.weight[2] = torch.tensor([11.0, 11.0])
        me.class_offset.weight[2] = torch.tensor([1.0, 1.0])

    # Sanity check: C is NOT included in D_overlap under gci0
    gci0_overlap = me(torch.tensor([[0, 1]]), "gci0")
    assert float(gci0_overlap.item()) > 1.0

    # Under ELBE's gci3 overlap formula, overlapping translated box gives 0, disjoint gives > 0
    loss_overlap = me(torch.tensor([[0, 0, 1]]), "gci3")
    loss_disjoint = me(torch.tensor([[0, 0, 2]]), "gci3")
    assert float(loss_overlap.item()) <= 1e-12
    assert float(loss_disjoint.item()) > 1.0


def test_effective_components() -> None:
    me = MultiELBEModule(
        nb_ont_classes=5,
        nb_rels=1,
        embed_dim=2,
        n_components=3,
    )
    with torch.no_grad():
        # Row 0: 3 disjoint non-degenerate boxes -> 3
        me.class_embed.weight[0] = torch.tensor([0.0, 0.0, 5.0, 5.0, 10.0, 10.0])
        me.class_offset.weight[0] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Row 1: comp 0 ([0, 2]^2) nested inside comp 1 ([-1, 3]^2), comp 2 ([10, 12]^2) separate -> 2
        me.class_embed.weight[1] = torch.tensor([1.0, 1.0, 1.0, 1.0, 11.0, 11.0])
        me.class_offset.weight[1] = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])

        # Row 2: comp 0 non-degenerate, comp 1 degenerate in one coord (0.0005 <= 1e-3), comp 2 nested in comp 0 -> 1
        me.class_embed.weight[2] = torch.tensor([0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
        me.class_offset.weight[2] = torch.tensor([2.0, 2.0, 1.0, 0.0005, 1.0, 1.0])

        # Row 3: all 3 components identical non-degenerate boxes -> 1
        me.class_embed.weight[3] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        me.class_offset.weight[3] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Row 4: all 3 components degenerate -> 0
        me.class_embed.weight[4] = torch.zeros(6)
        me.class_offset.weight[4] = torch.zeros(6)

    eff = me.effective_components(eps=1e-3)
    assert eff.shape == (5,)
    assert eff.tolist() == [3, 2, 1, 1, 0]


def test_component_aggregation_validation_and_soft_not_implemented() -> None:
    me = MultiELBEModule(
        nb_ont_classes=4,
        nb_rels=2,
        embed_dim=3,
        n_components=2,
        component_aggregation="hard",
        tau=0.5,
        diversity_reg=0.25,
    )
    assert me.component_aggregation == "hard"
    assert me.tau == 0.5
    assert me.diversity_reg == 0.25

    with pytest.raises(ValueError, match="component_aggregation"):
        MultiELBEModule(
            nb_ont_classes=4,
            nb_rels=2,
            embed_dim=3,
            component_aggregation="invalid",
        )

    with pytest.raises(NotImplementedError, match="implemented in a later task"):
        MultiELBEModule(
            nb_ont_classes=4,
            nb_rels=2,
            embed_dim=3,
            component_aggregation="soft",
        )
