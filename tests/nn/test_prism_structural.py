"""Unit tests for Prism structural losses and PrismModule structural objective."""
from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch as th

from mowl.nn import PrismModule
import mowl.nn.el.prism.losses as PL
from mowl.nn.el.structural import DefinitionTable, ExistenceTargets


def _supports_to_center_inradius(h: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """Invert support coordinates ``h = (h_1, h_2, h_3)`` to ``(center, inradius)``."""
    h1 = h[..., 0]
    h2 = h[..., 1]
    h3 = h[..., 2]
    r = (h1 + h2 + h3) / 3.0
    cx = (h2 - h3) / math.sqrt(3.0)
    cy = r - h1
    center = th.stack([cx, cy], dim=-1)
    return center, r.unsqueeze(-1)


def test_prism_nonempty_measure() -> None:
    """Empty triangle => negative; empty box coordinate => negative; both positive => positive."""
    # Row 0: triangle sum = 3.0 > 0, box w = (0.5, 1.0) -> min_m 2*w_m = 1.0 -> ne = min(3.0, 1.0) = 1.0
    # Row 1: triangle sum = -3.0 < 0, box w = (0.5, 1.0) -> ne = min(-3.0, 1.0) = -3.0 < 0
    # Row 2: triangle sum = 3.0 > 0, box w = (-0.25, 1.0) -> min_m 2*w_m = -0.5 -> ne = min(3.0, -0.5) = -0.5 < 0
    h = th.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=th.float64,
    )
    cen = th.zeros((3, 2), dtype=th.float64)
    w = th.tensor(
        [
            [0.5, 1.0],
            [0.5, 1.0],
            [-0.25, 1.0],
        ],
        dtype=th.float64,
    )
    r = PL.PrismRegions(h=h, cen=cen, w=w)
    ne = PL.prism_nonempty_measure(r)
    assert ne.shape == (3,)
    assert ne[0].item() == pytest.approx(1.0, abs=1e-12)
    assert ne[1].item() == pytest.approx(-3.0, abs=1e-12)
    assert ne[2].item() == pytest.approx(-0.5, abs=1e-12)
    assert ne[1].item() < 0.0
    assert ne[2].item() < 0.0


def test_prism_structural_zero_at_exact_t3() -> None:
    """Exact T3 prism configuration: all structural losses and the existence term are 0 at margin 0."""
    # Classes: 0=X1, 1=X2, 2=X3, 3=A (X1 cap X2), 4=B (X2 cap X3), 5=C (X1 cap X3), 6=TOP, 7=Z (primitive inside X1)
    # Include unmasked GCI0 rows (3, 6) [A <= TOP] and (7, 0) [Z <= X1] alongside the 6 definition rows
    gci0 = th.tensor(
        [[3, 0], [3, 1], [4, 1], [4, 2], [5, 0], [5, 2], [3, 6], [7, 0]],
        dtype=th.long,
    )
    gci1 = th.tensor([[0, 1, 3], [1, 2, 4], [0, 2, 5]], dtype=th.long)
    gci1_bot = th.tensor(
        [[3, 4], [4, 5], [3, 5], [3, 2], [4, 0], [5, 1]], dtype=th.long
    )
    gci0_bot = th.zeros((0, 2), dtype=th.long)

    mod = PrismModule(
        nb_ont_classes=8,
        nb_rels=1,
        embed_dim=4,
        margin=0.0,
        objective="structural",
        max_depth=8,
        existence_weight=1.0,
        size_reg=0.0,
    )
    h1 = th.tensor([3.0, 3.0, -1.0])
    h2 = th.tensor([3.0, -1.0, 3.0])
    h3 = th.tensor([-1.0, 3.0, 3.0])
    ha = th.minimum(h1, h2)
    hb = th.minimum(h2, h3)
    hc = th.minimum(h1, h3)
    htop = th.tensor([5.0, 5.0, 5.0])
    hz = ha.clone()  # non-empty (sum = 1.0) and strictly inside X1 (ha <= h1)
    all_h = th.stack([h1, h2, h3, ha, hb, hc, htop, hz], dim=0)
    centers, inradii = _supports_to_center_inradius(all_h)
    with th.no_grad():
        mod.tri_center.weight.copy_(centers)
        mod.tri_inradius.weight.copy_(inradii)
        mod.box_center.weight.zero_()
        mod.box_offset.weight.fill_(1.0)
        mod.box_offset.weight[7].fill_(0.5)

    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=8,
        k=1,
        max_components=2**30,
        top_idx=6,
        bottom_idx=None,
        enabled=True,
        max_level=8,
    )
    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=gci0_bot,
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=8,
        top_idx=6,
        bottom_idx=None,
    )
    mod.set_structural_tables(table, targets, top_idx=6)

    assert table.n_defined == 3
    assert table.def_mask.tolist() == [False, False, False, True, True, True, False, False]
    assert table.masked_gci0(gci0).tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]

    loss_0 = mod(gci0, "gci0")
    loss_1 = mod(gci1, "gci1")
    loss_1_bot = mod(gci1_bot, "gci1_bot")
    extra = mod.structural_extra_loss()

    assert th.all(loss_0 <= 1e-12)
    assert th.all(loss_1 == 0.0)
    assert th.all(loss_1_bot <= 1e-12)
    assert float(extra.item()) <= 1e-12

    # Displace Z (idx 7) outside X1 (idx 0) and verify the unmasked gci0 row (7, 0) scores > 0
    saved_box_c7 = mod.box_center.weight[7].clone()
    with th.no_grad():
        mod.box_center.weight[7].fill_(5.0)
    loss_0_displaced = mod(gci0, "gci0")
    assert float(loss_0_displaced[7].item()) > 0.0
    assert float(loss_0_displaced[6].item()) <= 1e-12
    with th.no_grad():
        mod.box_center.weight[7].copy_(saved_box_c7)

    # Also verify size_reg excludes top_idx (6) and averages |r| and 2|o| over primitive classes 0, 1, 2, 7
    mod.size_reg = 1.0
    extra_with_size = mod.structural_extra_loss()
    assert float(extra_with_size.item()) > 0.0
    # Mutating top_idx (6) parameters must not change structural_extra_loss
    with th.no_grad():
        mod.tri_inradius.weight[6].fill_(100.0)
        mod.box_offset.weight[6].fill_(100.0)
    assert float(mod.structural_extra_loss().item()) == pytest.approx(
        float(extra_with_size.item()), abs=1e-6
    )


def test_gci0_and_gci1_structural_hand_built_regions() -> None:
    """Direct unit tests on hand-built PrismRegions for gci0_structural and gci1_structural."""
    # 1. Included -> 0 at margin 0
    c_in = PL.PrismRegions(
        h=th.tensor([[1.0, 1.0, 1.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[0.5, 0.5]], dtype=th.float64),
    )
    d_big = PL.PrismRegions(
        h=th.tensor([[2.0, 2.0, 2.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[1.0, 1.0]], dtype=th.float64),
    )
    assert PL.gci0_structural(c_in, d_big, margin=0.0).item() == 0.0
    assert PL.gci1_structural(c_in, d_big, d_big, margin=0.0).item() == 0.0

    # 2. Not included -> > 0
    assert PL.gci0_structural(d_big, c_in, margin=0.0).item() > 0.0
    assert PL.gci1_structural(d_big, d_big, c_in, margin=0.0).item() > 0.0

    # 3. Empty C (gci0) or empty C cap D (gci1) -> 0 (vacuous) even when coordinates exceed target
    c_empty_tri = PL.PrismRegions(
        h=th.tensor([[10.0, -6.0, -6.0]], dtype=th.float64),  # sum = -2 < 0
        cen=th.tensor([[5.0, 5.0]], dtype=th.float64),
        w=th.tensor([[2.0, 2.0]], dtype=th.float64),
    )
    c_empty_box = PL.PrismRegions(
        h=th.tensor([[5.0, 5.0, 5.0]], dtype=th.float64),
        cen=th.tensor([[5.0, 5.0]], dtype=th.float64),
        w=th.tensor([[-0.1, 2.0]], dtype=th.float64),  # negative w_0 -> empty
    )
    assert PL.gci0_structural(c_empty_tri, c_in, margin=0.1).item() == 0.0
    assert PL.gci0_structural(c_empty_box, c_in, margin=0.1).item() == 0.0

    # Disjoint non-empty C and D -> empty C cap D -> vacuous 0 in gci1_structural
    d_disjoint_box = PL.PrismRegions(
        h=th.tensor([[1.0, 1.0, 1.0]], dtype=th.float64),
        cen=th.tensor([[10.0, 10.0]], dtype=th.float64),
        w=th.tensor([[0.5, 0.5]], dtype=th.float64),
    )
    e_tiny = PL.PrismRegions(
        h=th.tensor([[0.1, 0.1, 0.1]], dtype=th.float64),
        cen=th.tensor([[-5.0, -5.0]], dtype=th.float64),
        w=th.tensor([[0.1, 0.1]], dtype=th.float64),
    )
    assert PL.gci1_structural(c_in, d_disjoint_box, e_tiny, margin=0.1).item() == 0.0

    # 4. Inclusion into an EMPTY computed target -> > 0
    # Empty target in box (negative w) and empty target in triangle (sum(h) < 0)
    target_empty_box = PL.PrismRegions(
        h=th.tensor([[2.0, 2.0, 2.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[-1.0, 2.0]], dtype=th.float64),
    )
    target_empty_tri = PL.PrismRegions(
        h=th.tensor([[-1.0, -1.0, -1.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[2.0, 2.0]], dtype=th.float64),
    )
    assert PL.gci0_structural(c_in, target_empty_box, margin=0.0).item() > 0.0
    assert PL.gci0_structural(c_in, target_empty_tri, margin=0.0).item() > 0.0
    assert PL.gci1_structural(c_in, d_big, target_empty_box, margin=0.0).item() > 0.0
    assert PL.gci1_structural(c_in, d_big, target_empty_tri, margin=0.0).item() > 0.0

    # 5. Masked gci1 row -> exactly 0 with zero gradient to all inputs
    ch = th.tensor([[2.0, 2.0, 2.0]], dtype=th.float64, requires_grad=True)
    ccen = th.tensor([[1.0, 1.0]], dtype=th.float64, requires_grad=True)
    cw = th.tensor([[1.0, 1.0]], dtype=th.float64, requires_grad=True)
    dh = th.tensor([[2.0, 2.0, 2.0]], dtype=th.float64, requires_grad=True)
    dcen = th.tensor([[1.0, 1.0]], dtype=th.float64, requires_grad=True)
    dw = th.tensor([[1.0, 1.0]], dtype=th.float64, requires_grad=True)
    eh = th.tensor([[0.5, 0.5, 0.5]], dtype=th.float64, requires_grad=True)
    ecen = th.tensor([[0.0, 0.0]], dtype=th.float64, requires_grad=True)
    ew = th.tensor([[0.2, 0.2]], dtype=th.float64, requires_grad=True)

    c_grad = PL.PrismRegions(h=ch, cen=ccen, w=cw)
    d_grad = PL.PrismRegions(h=dh, cen=dcen, w=dw)
    e_grad = PL.PrismRegions(h=eh, cen=ecen, w=ew)
    # Without mask this violates inclusion (> 0)
    assert PL.gci1_structural(c_grad, d_grad, e_grad, margin=0.1).item() > 0.0
    masked_loss = PL.gci1_structural(
        c_grad, d_grad, e_grad, margin=0.1, masked=th.tensor([True])
    )
    assert masked_loss.item() == 0.0
    masked_loss.sum().backward()
    for tensor in (ch, ccen, cw, dh, dcen, dw, eh, ecen, ew):
        assert tensor.grad is not None
        assert th.all(tensor.grad == 0.0)

    # 6. At margin m > 0, an inclusion that holds with slack < m -> > 0
    # Here slack is 0.05 in both triangle supports and box half-widths; at m = 0.0 it is 0, at m = 0.1 > 0.05 it is > 0
    c_slack = PL.PrismRegions(
        h=th.tensor([[0.95, 0.95, 0.95]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[0.95, 0.95]], dtype=th.float64),
    )
    d_slack = PL.PrismRegions(
        h=th.tensor([[1.0, 1.0, 1.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[1.0, 1.0]], dtype=th.float64),
    )
    assert PL.gci0_structural(c_slack, d_slack, margin=0.0).item() == 0.0
    assert PL.gci0_structural(c_slack, d_slack, margin=0.1).item() > 0.0
    assert PL.gci1_structural(c_slack, d_slack, d_slack, margin=0.0).item() == 0.0
    assert PL.gci1_structural(c_slack, d_slack, d_slack, margin=0.1).item() > 0.0

    # Also check prism_regions_intersection helper directly
    inter = PL.prism_regions_intersection(c_in, d_big)
    assert th.allclose(inter.h, c_in.h)
    assert th.allclose(inter.cen, c_in.cen)
    assert th.allclose(inter.w, c_in.w)


def test_prism_gci1_bot_triangle_separation() -> None:
    """Box parts overlap, triangles separated => gci1_bot_structural is 0."""
    # h_c = (1, 1, -1) (sum = 1 > 0), h_d = (-1, -1, 3) (sum = 1 > 0)
    # min(h_c, h_d) = (-1, -1, -1) -> sum = -3.0
    c = PL.PrismRegions(
        h=th.tensor([[1.0, 1.0, -1.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[1.0, 1.0]], dtype=th.float64),
    )
    d = PL.PrismRegions(
        h=th.tensor([[-1.0, -1.0, 3.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[1.0, 1.0]], dtype=th.float64),
    )
    loss_zero = PL.gci1_bot_structural(c, d, margin=0.5)
    assert loss_zero.shape == (1,)
    assert loss_zero.item() == 0.0

    # When both triangles and boxes overlap: h_c = h_d = (1, 1, 1) (sum = 3), box w = 1 -> ne = 2.0
    c_ov = PL.PrismRegions(
        h=th.tensor([[1.0, 1.0, 1.0]], dtype=th.float64),
        cen=th.tensor([[0.0, 0.0]], dtype=th.float64),
        w=th.tensor([[1.0, 1.0]], dtype=th.float64),
    )
    loss_pos = PL.gci1_bot_structural(c_ov, c_ov, margin=0.1)
    assert loss_pos.item() == pytest.approx((2.0 + 0.1) ** 2, abs=1e-12)


def test_prism_definition_depth_cap() -> None:
    """Recursive prism_intersection respects max_depth cap on DefinitionTable and matches closed-form intersection."""
    # 0:A, 1:Aux, 2:X1, 3:X2, 4:X3
    # Aux == X1 cap X2 (level 1), A == Aux cap X3 (level 2)
    gci0 = th.tensor([[1, 2], [1, 3], [0, 1], [0, 4]], dtype=th.long)
    gci1 = th.tensor([[2, 3, 1], [1, 4, 0]], dtype=th.long)

    mod = PrismModule(
        nb_ont_classes=5,
        nb_rels=1,
        embed_dim=4,
        margin=0.0,
        objective="structural",
        max_depth=2,
    )
    # Set distinct boxes on X1, X2, X3 so Aux has positive width in dim 0 and negative width in dim 1
    with th.no_grad():
        # X1 (idx 2): box [0, 4] x [0, 2] -> cen=(2, 1), w=(2, 1)
        mod.box_center.weight[2] = th.tensor([2.0, 1.0])
        mod.box_offset.weight[2] = th.tensor([2.0, 1.0])
        # X2 (idx 3): box [1, 5] x [3, 5] -> cen=(3, 4), w=(2, 1)
        # X1 cap X2: [1, 4] x [3, 2] -> cen=(2.5, 2.5), w=(1.5, -0.5)
        mod.box_center.weight[3] = th.tensor([3.0, 4.0])
        mod.box_offset.weight[3] = th.tensor([2.0, 1.0])
        # X3 (idx 4): box [2, 6] x [0, 6] -> cen=(4, 3), w=(2, 3)
        # (X1 cap X2) cap X3: [2, 4] x [3, 2] -> cen=(3.0, 2.5), w=(1.0, -0.5)
        mod.box_center.weight[4] = th.tensor([4.0, 3.0])
        mod.box_offset.weight[4] = th.tensor([2.0, 3.0])

    t_depth2 = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=1,
        max_components=2**30,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
        max_level=2,
    )
    targets = ExistenceTargets(
        classes=th.arange(5, dtype=th.long),
        pairs=th.zeros((0, 2), dtype=th.long),
    )
    mod.set_structural_tables(t_depth2, targets)
    assert t_depth2.def_mask.tolist() == [True, True, False, False, False]

    reg = mod.class_regions(th.tensor([0, 1, 2, 3, 4], dtype=th.long))
    # Aux (idx 1) has signed w = (1.5, -0.5)
    assert th.allclose(reg.cen[1], th.tensor([2.5, 2.5]), atol=1e-6)
    assert th.allclose(reg.w[1], th.tensor([1.5, -0.5]), atol=1e-6)
    # A (idx 0) has signed w = (1.0, -0.5) (preserved negative width, NOT abs'd!)
    assert th.allclose(reg.cen[0], th.tensor([3.0, 2.5]), atol=1e-6)
    assert th.allclose(reg.w[0], th.tensor([1.0, -0.5]), atol=1e-6)
    assert th.allclose(
        reg.h[0], th.minimum(th.minimum(reg.h[2], reg.h[3]), reg.h[4]), atol=1e-6
    )

    all_reg = mod.all_class_regions()
    assert not all_reg.h.requires_grad
    assert all_reg.h.device.type == "cpu"
    assert th.allclose(all_reg.w, reg.w.detach().cpu(), atol=1e-6)

    # With max_level=1, A (level 2) is rejected and stays primitive
    t_depth1 = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=1,
        max_components=2**30,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
        max_level=1,
    )
    assert t_depth1.def_mask.tolist() == [False, True, False, False, False]

    # Install t_depth1 and assert class_regions returns the primitive prism for the rejected level-2 class A (idx 0)
    mod.set_structural_tables(t_depth1, targets)
    reg_d1 = mod.class_regions(th.tensor([0, 1], dtype=th.long))
    prim_h0, prim_bc0, prim_bo0 = mod.prism_params(th.tensor([0], dtype=th.long))
    assert th.allclose(reg_d1.h[0:1], prim_h0, atol=1e-6)
    assert th.allclose(reg_d1.cen[0:1], prim_bc0, atol=1e-6)
    assert th.allclose(reg_d1.w[0:1], prim_bo0, atol=1e-6)
    # While Aux (idx 1, level 1) is still computed as X1 cap X2
    assert th.allclose(reg_d1.cen[1], th.tensor([2.5, 2.5]), atol=1e-6)
    assert th.allclose(reg_d1.w[1], th.tensor([1.5, -0.5]), atol=1e-6)


def test_role_and_assertion_prisms_delegation() -> None:
    """*_prisms variants produce identical values to existing embedding-based functions."""
    n_classes, n_rels, n_inds, d, B = 8, 3, 4, 5, 12
    mod = PrismModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        margin=0.1,
    )
    gen = th.Generator().manual_seed(7)
    gci2_batch = th.cat(
        [
            th.randint(0, n_classes, (B, 1), generator=gen),
            th.randint(0, n_rels, (B, 1), generator=gen),
            th.randint(0, n_classes, (B, 1), generator=gen),
        ],
        dim=1,
    )
    gci3_batch = th.cat(
        [
            th.randint(0, n_rels, (B, 1), generator=gen),
            th.randint(0, n_classes, (B, 1), generator=gen),
            th.randint(0, n_classes, (B, 1), generator=gen),
        ],
        dim=1,
    )
    ca_batch = th.cat(
        [
            th.randint(0, n_inds, (B, 1), generator=gen),
            th.randint(0, n_classes, (B, 1), generator=gen),
        ],
        dim=1,
    )

    rc_gci2 = mod.class_regions(gci2_batch[:, 0])
    rd_gci2 = mod.class_regions(gci2_batch[:, 2])
    rel_gci2 = mod.rel_embed(gci2_batch[:, 1])

    assert th.allclose(
        PL.gci2_score_prisms(rc_gci2, rd_gci2, rel_gci2, mod.margin),
        mod.gci2_loss(gci2_batch, neg=False),
        atol=1e-6,
    )
    assert th.allclose(
        PL.gci2_score_neg_prisms(rc_gci2, rd_gci2, rel_gci2, mod.margin),
        mod.gci2_loss(gci2_batch, neg=True),
        atol=1e-6,
    )

    rc_gci3 = mod.class_regions(gci3_batch[:, 1])
    rd_gci3 = mod.class_regions(gci3_batch[:, 2])
    rel_gci3 = mod.rel_embed(gci3_batch[:, 0])
    assert th.allclose(
        PL.gci3_score_prisms(rc_gci3, rd_gci3, rel_gci3, mod.margin),
        mod.gci3_loss(gci3_batch, neg=False),
        atol=1e-6,
    )
    assert th.allclose(
        PL.gci3_bot_score_prisms(rc_gci3, mod.margin),
        mod.gci3_bot_loss(gci3_batch, neg=False),
        atol=1e-6,
    )

    ind_pts = mod.ind_embed(ca_batch[:, 0])
    rc_ca = mod.class_regions(ca_batch[:, 1])
    assert th.allclose(
        PL.class_assertion_score_prisms(ind_pts, rc_ca, mod.margin),
        mod.class_assertion_loss(ca_batch, neg=False),
        atol=1e-6,
    )
