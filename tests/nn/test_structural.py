"""Unit tests for mowl.nn.el.structural (region algebra, structural losses, tables)."""
from __future__ import annotations

import time
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
from mowl.nn.el.structural import (
    DefinitionTable,
    ExistenceTargets,
    Regions,
    WitnessTargets,
    existence_loss,
    gci0_bot_structural,
    gci0_structural,
    gci1_bot_structural,
    gci1_structural,
    inclusion_tight,
    inside_measure,
    nonempty_measure,
    pair_size,
    pair_slots,
    region_size,
    signed_overlap,
    slot_components,
    witness_loss,
)
from tests.geometric.helpers import NS, fixture_path, t3_cm, t3_dataset


def test_signed_overlap_sign() -> None:
    """Overlapping intervals give the intersection length; separated give the negative gap."""
    # Batch of 1, ka=2, kb=2, d=1 in float64
    # A_0 = [0, 2] (c=1, w=1), A_1 = [0, 1] (c=0.5, w=0.5)
    # B_0 = [1.5, 3.5] (c=2.5, w=1), B_1 = [3, 5] (c=4, w=1)
    cen_a = th.tensor([[[1.0], [0.5]]], dtype=th.float64)
    w_a = th.tensor([[[1.0], [0.5]]], dtype=th.float64)
    cen_b = th.tensor([[[2.5], [4.0]]], dtype=th.float64)
    w_b = th.tensor([[[1.0], [1.0]]], dtype=th.float64)

    sov = signed_overlap(cen_a, w_a, cen_b, w_b)
    assert sov.shape == (1, 2, 2, 1)
    # A_0 [0, 2] and B_0 [1.5, 3.5] overlap on [1.5, 2.0] -> length +0.5
    assert sov[0, 0, 0, 0].item() == pytest.approx(0.5, abs=1e-12)
    # A_0 [0, 2] and B_1 [3, 5] separated by gap 1.0 -> -1.0
    assert sov[0, 0, 1, 0].item() == pytest.approx(-1.0, abs=1e-12)
    # A_1 [0, 1] and B_0 [1.5, 3.5] separated by gap 0.5 -> -0.5
    assert sov[0, 1, 0, 0].item() == pytest.approx(-0.5, abs=1e-12)
    # A_1 [0, 1] and B_1 [3, 5] separated by gap 2.0 -> -2.0
    assert sov[0, 1, 1, 0].item() == pytest.approx(-2.0, abs=1e-12)


def test_pair_slots_level_two() -> None:
    """X (k=2), Y (k=2) -> 4 pair slots; with one inactive parent slot only 2 active and they come first; w negative in disjoint coordinate."""
    # 2-D boxes, B=1, k=2
    # X_0: x in [0, 2], y in [0, 2] -> cen=(1, 1), w=(1, 1), active=False
    # X_1: x in [0, 2], y in [0, 2] -> cen=(1, 1), w=(1, 1), active=True
    cen_x = th.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=th.float64)
    w_x = th.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=th.float64)
    act_x_all = th.tensor([[True, True]], dtype=th.bool)

    # Y_0: x in [1, 3] (overlaps X in [1,2]), y in [3, 5] (disjoint from [0,2] by gap 1)
    # Y_1: x in [0.5, 1.5], y in [0.5, 1.5] (overlaps X in both)
    cen_y = th.tensor([[[2.0, 4.0], [1.0, 1.0]]], dtype=th.float64)
    w_y = th.tensor([[[1.0, 1.0], [0.5, 0.5]]], dtype=th.float64)
    act_y_all = th.tensor([[True, True]], dtype=th.bool)

    # Case 1: all 4 parent slots active -> 4 active pair slots
    rx_all = Regions(cen=cen_x, w=w_x, active=act_x_all)
    ry_all = Regions(cen=cen_y, w=w_y, active=act_y_all)
    pairs_4 = pair_slots(rx_all, ry_all, max_slots=4)
    assert pairs_4.cen.shape == (1, 4, 2)
    assert pairs_4.w.shape == (1, 4, 2)
    assert pairs_4.active.tolist() == [[True, True, True, True]]
    # Pair (0, 0): x-inter [1, 2] -> cen=1.5, w=0.5; y-inter lower=3, upper=2 -> cen=2.5, w=-0.5 (< 0)
    assert pairs_4.cen[0, 0].tolist() == pytest.approx([1.5, 2.5], abs=1e-12)
    assert pairs_4.w[0, 0].tolist() == pytest.approx([0.5, -0.5], abs=1e-12)
    assert pairs_4.w[0, 0, 1].item() < 0.0

    # Case 2: X_0 inactive -> only pairs (1, 0) and (1, 1) are active, and they must come first
    act_x_one = th.tensor([[False, True]], dtype=th.bool)
    rx_one = Regions(cen=cen_x, w=w_x, active=act_x_one)
    pairs_partial = pair_slots(rx_one, ry_all, max_slots=6)
    assert pairs_partial.cen.shape == (1, 6, 2)
    assert pairs_partial.active.tolist() == [[True, True, False, False, False, False]]
    # Active slot 0 is pair (1, 0): cen=(1.5, 2.5), w=(0.5, -0.5)
    assert pairs_partial.cen[0, 0].tolist() == pytest.approx([1.5, 2.5], abs=1e-12)
    assert pairs_partial.w[0, 0].tolist() == pytest.approx([0.5, -0.5], abs=1e-12)
    # Active slot 1 is pair (1, 1): cen=(1.0, 1.0), w=(0.5, 0.5)
    assert pairs_partial.cen[0, 1].tolist() == pytest.approx([1.0, 1.0], abs=1e-12)
    assert pairs_partial.w[0, 1].tolist() == pytest.approx([0.5, 0.5], abs=1e-12)
    # Inactive padding slots have cen=0, w=-1
    assert th.all(pairs_partial.cen[0, 2:] == 0.0)
    assert th.all(pairs_partial.w[0, 2:] == -1.0)

    # Exceeding max_slots raises ValueError
    with pytest.raises(ValueError):
        pair_slots(rx_all, ry_all, max_slots=3)


def test_gci1_bot_single_coordinate_separation() -> None:
    """2-D boxes overlapping in x, separated by 0.1 in y -> loss 0 at margin 0.05; overlapping in both -> loss > 0."""
    # C: x in [0, 2], y in [0, 1] -> cen=(1.0, 0.5), w=(1.0, 0.5)
    # D_sep: x in [0.5, 1.5] (overlap 1.0), y in [1.1, 2.1] (separated by 0.1 in y)
    c = Regions(
        cen=th.tensor([[[1.0, 0.5]]], dtype=th.float64),
        w=th.tensor([[[1.0, 0.5]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    d_sep = Regions(
        cen=th.tensor([[[1.0, 1.6]]], dtype=th.float64),
        w=th.tensor([[[0.5, 0.5]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    loss_sep = gci1_bot_structural(c, d_sep, margin=0.05)
    assert loss_sep.shape == (1,)
    assert loss_sep.item() == pytest.approx(0.0, abs=1e-12)

    # D_ov: x in [0.5, 1.5] (overlap 1.0), y in [0.8, 1.8] (overlap 0.2 in y)
    d_ov = Regions(
        cen=th.tensor([[[1.0, 1.3]]], dtype=th.float64),
        w=th.tensor([[[0.5, 0.5]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    loss_ov = gci1_bot_structural(c, d_ov, margin=0.05)
    assert loss_ov.item() > 0.0
    # ne = min(1.0, 0.2) = 0.2; relu(0.2 + 0.05)^2 = 0.25^2 = 0.0625
    assert loss_ov.item() == pytest.approx(0.25**2, abs=1e-12)


def test_gci1_vacuous_when_empty() -> None:
    """C cap D empty -> gci1_structural == 0."""
    c = Regions(
        cen=th.tensor([[[0.0]]], dtype=th.float64),
        w=th.tensor([[[0.5]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    d = Regions(
        cen=th.tensor([[[3.0]]], dtype=th.float64),
        w=th.tensor([[[0.5]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    # E is far away at [10, 11], yet C cap D is empty so gci1 loss is vacuously 0
    e = Regions(
        cen=th.tensor([[[10.5]]], dtype=th.float64),
        w=th.tensor([[[0.5]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    masked = th.tensor([False])
    loss = gci1_structural(c, d, e, margin=0.05, masked=masked)
    assert loss.shape == (1,)
    assert loss.item() == 0.0


def test_gci1_masked_row_zero() -> None:
    """masked=True -> exactly 0 even when violated."""
    # Row 0 (masked=True) and Row 1 (masked=False), both with C = D = [0, 2] not inside E = [5, 6]
    c = Regions(
        cen=th.tensor([[[1.0]], [[1.0]]], dtype=th.float64),
        w=th.tensor([[[1.0]], [[1.0]]], dtype=th.float64),
        active=th.tensor([[True], [True]]),
    )
    d = Regions(
        cen=th.tensor([[[1.0]], [[1.0]]], dtype=th.float64),
        w=th.tensor([[[1.0]], [[1.0]]], dtype=th.float64),
        active=th.tensor([[True], [True]]),
    )
    e = Regions(
        cen=th.tensor([[[5.5]], [[5.5]]], dtype=th.float64),
        w=th.tensor([[[0.5]], [[0.5]]], dtype=th.float64),
        active=th.tensor([[True], [True]]),
    )
    masked = th.tensor([True, False], dtype=th.bool)
    loss = gci1_structural(c, d, e, margin=0.0, masked=masked)
    assert loss.shape == (2,)
    assert loss[0].item() == 0.0
    assert loss[1].item() > 0.0


def test_existence_gradient_to_nearest_pair() -> None:
    """X and Y each with two separated components; pair_size has non-zero gradient on exactly the nearest pair's parameters."""
    # 1-D, k=2:
    # X_0 = [0, 1] (c=0.5, w=0.5), X_1 = [10, 11] (c=10.5, w=0.5)
    # Y_0 = [20, 21] (c=20.5, w=0.5), Y_1 = [12, 13] (c=12.5, w=0.5)
    # Gaps:
    # (X_0, Y_0): [0,1] vs [20,21] -> gap 19 (sov = -19)
    # (X_0, Y_1): [0,1] vs [12,13] -> gap 11 (sov = -11)
    # (X_1, Y_0): [10,11] vs [20,21] -> gap 9 (sov = -9)
    # (X_1, Y_1): [10,11] vs [12,13] -> gap 1 (sov = -1)  <-- nearest pair is (1, 1)
    cen_x = th.tensor([[[0.5], [10.5]]], dtype=th.float64, requires_grad=True)
    w_x = th.tensor([[[0.5], [0.5]]], dtype=th.float64, requires_grad=True)
    cen_y = th.tensor([[[20.5], [12.5]]], dtype=th.float64, requires_grad=True)
    w_y = th.tensor([[[0.5], [0.5]]], dtype=th.float64, requires_grad=True)

    rx = Regions(cen=cen_x, w=w_x, active=th.tensor([[True, True]]))
    ry = Regions(cen=cen_y, w=w_y, active=th.tensor([[True, True]]))

    ps = pair_size(rx, ry)
    assert ps.shape == (1,)
    assert ps.item() == pytest.approx(-1.0, abs=1e-12)

    loss = existence_loss(
        th.empty(0, dtype=th.float64), ps, margin=0.1
    )
    loss.backward()

    assert cen_x.grad is not None and w_x.grad is not None
    assert cen_y.grad is not None and w_y.grad is not None
    # Non-nearest components X_0 and Y_0 get exact 0 gradient
    assert cen_x.grad[0, 0, 0].item() == 0.0
    assert w_x.grad[0, 0, 0].item() == 0.0
    assert cen_y.grad[0, 0, 0].item() == 0.0
    assert w_y.grad[0, 0, 0].item() == 0.0
    # Nearest pair (X_1, Y_1) gets non-zero gradient pulling X_1 right and Y_1 left and growing widths
    assert cen_x.grad[0, 1, 0].item() < 0.0
    assert cen_y.grad[0, 1, 0].item() > 0.0
    assert w_x.grad[0, 1, 0].item() < 0.0
    assert w_y.grad[0, 1, 0].item() < 0.0


def test_inclusion_tight_zero_iff_subset() -> None:
    """inclusion_tight at margin 0 is 0 iff box A is coordinatewise a subset of box B."""
    # B = [0, 4] x [0, 4] -> cen=(2, 2), w=(2, 2)
    cen_b = th.tensor([[[2.0, 2.0]]], dtype=th.float64)
    w_b = th.tensor([[[2.0, 2.0]]], dtype=th.float64)

    # A_sub = [1, 3] x [0, 4] (inside B), A_out = [1, 4.5] x [1, 3] (exceeds B by 0.5 in x)
    cen_a = th.tensor([[[2.0, 2.0], [2.75, 2.0]]], dtype=th.float64)
    w_a = th.tensor([[[1.0, 2.0], [1.75, 1.0]]], dtype=th.float64)

    scores = inclusion_tight(cen_a, w_a, cen_b, w_b, margin=0.0)
    assert scores.shape == (1, 2, 1)
    assert scores[0, 0, 0].item() == pytest.approx(0.0, abs=1e-12)
    assert scores[0, 1, 0].item() == pytest.approx(0.5, abs=1e-12)

    # With positive margin=0.1, A_sub (touching boundary in y) gets score 0.1
    scores_m = inclusion_tight(cen_a, w_a, cen_b, w_b, margin=0.1)
    assert scores_m[0, 0, 0].item() == pytest.approx(0.1, abs=1e-12)


def test_definition_table_t3() -> None:
    """gci0/gci1 of T3 with k=2, K=16 -> def_mask true for A, B, C, comps = 4, masked rows exactly the 6 gci0 and 3 gci1 definition rows."""
    # Indices: 0:A, 1:B, 2:C, 3:X1, 4:X2, 5:X3, 6:Z, 7:TOP, 8:BOT
    gci0 = th.tensor(
        [
            [0, 3],  # A <= X1
            [0, 4],  # A <= X2
            [1, 4],  # B <= X2
            [1, 5],  # B <= X3
            [2, 3],  # C <= X1
            [2, 5],  # C <= X3
            [6, 3],  # Z <= X1 (non-definition row)
        ],
        dtype=th.long,
    )
    gci1 = th.tensor(
        [
            [3, 4, 0],  # X1 cap X2 <= A
            [4, 5, 1],  # X2 cap X3 <= B
            [3, 5, 2],  # X1 cap X3 <= C
            [3, 6, 4],  # X1 cap Z <= X2 (not a definition)
        ],
        dtype=th.long,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=9,
        k=2,
        max_components=16,
        top_idx=7,
        bottom_idx=8,
        enabled=True,
    )
    assert table.n_defined == 3
    assert table.max_level == 1
    assert table.def_mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert table.def_x[:3].tolist() == [3, 4, 3]
    assert table.def_y[:3].tolist() == [4, 5, 5]
    assert table.def_x[3:].tolist() == [-1] * 6
    assert table.def_y[3:].tolist() == [-1] * 6
    assert table.level.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert table.comps.tolist() == [4, 4, 4, 2, 2, 2, 2, 2, 2]

    m0 = table.masked_gci0(gci0)
    assert m0.tolist() == [True, True, True, True, True, True, False]

    m1 = table.masked_gci1(gci1)
    assert m1.tolist() == [True, True, True, False]
    # Reversed order (X2, X1, A) is also masked
    assert table.masked_gci1(th.tensor([[4, 3, 0]], dtype=th.long)).tolist() == [True]


def test_definition_table_two_levels_and_cap() -> None:
    """Aux == X1 cap X2, A == Aux cap X3, k=2: K=16 -> both accepted (level(A)=2, comps(A)=8); K=4 -> only Aux accepted."""
    # Put A at index 0 and Aux at index 1 (as in alphabetical IRI order) to also test dependency order
    # 0:A, 1:Aux, 2:X1, 3:X2, 4:X3
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

    t16 = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    assert t16.n_defined == 2
    assert t16.def_mask.tolist() == [True, True, False, False, False]
    assert t16.level.tolist() == [2, 1, 0, 0, 0]
    assert t16.comps.tolist() == [8, 4, 2, 2, 2]
    assert t16.max_level == 2

    t4 = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=4,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    assert t4.n_defined == 1
    assert t4.def_mask.tolist() == [False, True, False, False, False]
    assert t4.level.tolist() == [0, 1, 0, 0, 0]
    assert t4.comps.tolist() == [2, 4, 2, 2, 2]
    assert t4.max_level == 1
    assert t4.masked_gci0(gci0).tolist() == [True, True, False, False]
    assert t4.masked_gci1(gci1).tolist() == [True, False]

    # Also check max_level=1 cap directly even with max_components=16
    t_lvl1 = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
        max_level=1,
    )
    assert t_lvl1.def_mask.tolist() == [False, True, False, False, False]


def test_definition_table_rejects_cycle() -> None:
    """A == B cap C, B == A cap D -> lowest-index cycle node (A) rejected, B accepted."""
    # 0:A, 1:B, 2:C, 3:D
    gci0 = th.tensor(
        [
            [0, 1],  # A <= B
            [0, 2],  # A <= C
            [1, 0],  # B <= A
            [1, 3],  # B <= D
        ],
        dtype=th.long,
    )
    gci1 = th.tensor(
        [
            [1, 2, 0],  # B cap C <= A
            [0, 3, 1],  # A cap D <= B
        ],
        dtype=th.long,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=4,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    assert table.n_defined == 1
    assert table.def_mask.tolist() == [False, True, False, False]
    assert table.def_x[1].item() == 0
    assert table.def_y[1].item() == 3
    assert table.level.tolist() == [0, 1, 0, 0]
    assert table.comps.tolist() == [2, 4, 2, 2]


def test_definition_table_disabled() -> None:
    """enabled=False -> nothing defined, nothing masked."""
    gci0 = th.tensor([[0, 2], [0, 3]], dtype=th.long)
    gci1 = th.tensor([[2, 3, 0]], dtype=th.long)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=4,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=False,
    )
    assert table.n_defined == 0
    assert table.max_level == 0
    assert not bool(table.def_mask.any())
    assert table.comps.tolist() == [2, 2, 2, 2]
    assert not bool(table.masked_gci0(gci0).any())
    assert not bool(table.masked_gci1(gci1).any())


def test_definition_table_role_auxiliary_pizza_style() -> None:
    """Pizza-style normal forms (A <= X, A <= exists r.C, exists r.C <= Aux, X cap Aux <= A)
    -> A defined with definers (X, Aux), masked rows = {(A,X) gci0, (X,Aux,A)/(Aux,X,A) gci1}
    and gci2/gci3 rows are NOT masked.
    """
    # Indices: 0:A, 1:X, 2:C, 3:Aux, 4:Other
    # Normal forms of A == X cap exists r.C:
    # gci0: A <= X (0, 1), plus an unrelated row Other <= X (4, 1)
    # gci2: A <= exists r0.C (0, 0, 2)
    # gci3: exists r0.C <= Aux (0, 2, 3)
    # gci1: X cap Aux <= A (1, 3, 0), plus an unrelated row C cap Other <= X (2, 4, 1)
    gci0 = th.tensor([[0, 1], [4, 1]], dtype=th.long)
    gci1 = th.tensor([[1, 3, 0], [2, 4, 1]], dtype=th.long)
    gci2 = th.tensor([[0, 0, 2]], dtype=th.long)
    gci3 = th.tensor([[0, 2, 3]], dtype=th.long)

    # Without gci2/gci3, A is not detected because (A, Aux) is not in gci0
    table_no_roles = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    assert table_no_roles.n_defined == 0

    # Mismatched role or filler in gci3 does not trigger definition
    table_mismatch = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
        gci2=gci2,
        gci3=th.tensor([[1, 2, 3]], dtype=th.long),
    )
    assert table_mismatch.n_defined == 0

    # With matching gci2 (A, r, C) and gci3 (r, C, Aux), A is defined with definers (X, Aux)
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=5,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
        gci2=gci2,
        gci3=gci3,
    )
    assert table.n_defined == 1
    assert table.max_level == 1
    assert table.def_mask.tolist() == [True, False, False, False, False]
    assert table.def_x[0].item() == 1
    assert table.def_y[0].item() == 3
    assert table.level.tolist() == [1, 0, 0, 0, 0]
    assert table.comps.tolist() == [4, 2, 2, 2, 2]

    # Masked gci0 rows: (A, X) is masked, (Other, X) is not
    assert table.masked_gci0(gci0).tolist() == [True, False]
    # Masked gci1 rows: both (X, Aux, A) and (Aux, X, A) are masked, (C, Other, X) is not
    assert table.masked_gci1(gci1).tolist() == [True, False]
    assert table.masked_gci1(th.tensor([[3, 1, 0]], dtype=th.long)).tolist() == [True]
    # gci2 and gci3 rows are never masked by DefinitionTable
    assert not hasattr(table, "masked_gci2")
    assert not hasattr(table, "masked_gci3")


def test_existence_targets_asserted_rule() -> None:
    """Without cm: classes without gci0_bot and gci1 pairs without gci1_bot (excluding top/bot)."""
    # Classes: 0:A, 1:B, 2:C, 3:Unsat, 4:TOP, 5:BOT
    gci0_bot = th.tensor([[3, 5]], dtype=th.long)
    gci1 = th.tensor(
        [
            [0, 1, 2],  # (0, 1) -> in gci1_bot -> excluded
            [1, 0, 2],  # duplicate reversed
            [0, 2, 1],  # (0, 2) -> kept
            [2, 2, 0],  # C == D -> excluded
            [0, 4, 1],  # involves TOP -> excluded
        ],
        dtype=th.long,
    )
    gci1_bot = th.tensor([[1, 0]], dtype=th.long)

    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=gci0_bot,
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=6,
        top_idx=4,
        bottom_idx=5,
    )
    assert targets.classes.tolist() == [0, 1, 2]
    assert targets.pairs.tolist() == [[0, 2]]


def test_existence_targets_with_cm() -> None:
    """With T3 canonical model: pairs (X1,X2), (X2,X3), (X1,X3) present, (A,B) absent."""
    ds = PathDataset(fixture_path("t3.owl"))
    cm = CanonicalModel.from_dataset(ds)

    names = ["A", "B", "C", "X1", "X2", "X3", "Z"]
    class_index_dict = {NS + name: i for i, name in enumerate(names)}
    top_idx = 7
    bot_idx = 8
    class_index_dict["http://www.w3.org/2002/07/owl#Thing"] = top_idx
    class_index_dict["http://www.w3.org/2002/07/owl#Nothing"] = bot_idx
    index_to_iri = {idx: iri for iri, idx in class_index_dict.items()}

    # Include (X1,X2), (X2,X3), (X1,X3) and also (A,B) in gci1 to verify cm filters (A,B) out
    gci0_bot = th.zeros((0, 2), dtype=th.long)
    gci1 = th.tensor(
        [
            [3, 4, 0],  # (X1, X2)
            [4, 5, 1],  # (X2, X3)
            [3, 5, 2],  # (X1, X3)
            [0, 1, 6],  # (A, B) -> unsatisfiable in T3 cm
        ],
        dtype=th.long,
    )
    gci1_bot = th.zeros((0, 2), dtype=th.long)

    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=gci0_bot,
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=9,
        top_idx=top_idx,
        bottom_idx=bot_idx,
        index_to_iri=index_to_iri,
        cm=cm,
    )
    assert targets.classes.tolist() == [0, 1, 2, 3, 4, 5, 6]
    pair_set = {tuple(row) for row in targets.pairs.tolist()}
    assert pair_set == {(3, 4), (4, 5), (3, 5)}
    assert (0, 1) not in pair_set


def test_region_size_and_gci0_structural_and_float32() -> None:
    """region_size requires >= 1 active slot; gci0_structural and gci0_bot_structural work in float64 and float32."""
    for dtype in (th.float64, th.float32):
        # Slot 0 active with w = (0.5, 1.0) -> ne = 1.0; Slot 1 inactive with w = (5.0, 5.0)
        r = Regions(
            cen=th.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=dtype),
            w=th.tensor([[[0.5, 1.0], [5.0, 5.0]]], dtype=dtype),
            active=th.tensor([[True, False]]),
        )
        ne = nonempty_measure(r.w)
        assert ne.shape == (1, 2)
        rs = region_size(r)
        assert rs.shape == (1,)
        assert rs.item() == pytest.approx(1.0, abs=1e-6)

        # gci0_bot_structural averages relu(ne)^2 over active slots only
        bot_loss = gci0_bot_structural(r)
        assert bot_loss.shape == (1,)
        assert bot_loss.item() == pytest.approx(1.0, abs=1e-6)

        # gci0_structural: if C has only empty active slots (w < 0), loss is 0
        c_empty = Regions(
            cen=th.tensor([[[0.0, 0.0]]], dtype=dtype),
            w=th.tensor([[[-0.5, 1.0]]], dtype=dtype),
            active=th.tensor([[True]]),
        )
        assert gci0_structural(c_empty, r, margin=0.05).item() == 0.0
        # C = r inside D = [ -2, 2 ]^2 -> loss 0 at margin 0
        d_big = Regions(
            cen=th.tensor([[[0.0, 0.0]]], dtype=dtype),
            w=th.tensor([[[2.0, 2.0]]], dtype=dtype),
            active=th.tensor([[True]]),
        )
        assert gci0_structural(r, d_big, margin=0.0).item() == 0.0

    # All slots inactive in a row -> ValueError
    r_bad = Regions(
        cen=th.zeros((1, 2, 2), dtype=th.float64),
        w=th.ones((1, 2, 2), dtype=th.float64),
        active=th.tensor([[False, False]]),
    )
    with pytest.raises(ValueError):
        region_size(r_bad)


def test_definition_table_scale_linear() -> None:
    """20 000 independent candidates build in well under 5 s and produce n_defined == 20000."""
    n_defs = 20_000
    x_idx = n_defs
    y_idx = n_defs + 1
    n_classes = n_defs + 2

    a_indices = th.arange(n_defs, dtype=th.long)
    x_col = th.full((n_defs,), x_idx, dtype=th.long)
    y_col = th.full((n_defs,), y_idx, dtype=th.long)

    gci0 = th.cat(
        [
            th.stack([a_indices, x_col], dim=1),
            th.stack([a_indices, y_col], dim=1),
        ],
        dim=0,
    )
    gci1 = th.stack([x_col, y_col, a_indices], dim=1)

    t0 = time.perf_counter()
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=n_classes,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"DefinitionTable took {elapsed:.2f}s (>= 5.0s)"
    assert table.n_defined == n_defs
    assert table.max_level == 1
    assert bool(table.def_mask[:n_defs].all())
    assert not bool(table.def_mask[n_defs:].any())


def test_definition_table_cycle_with_smaller_acyclic_candidate() -> None:
    """Cycle where the lowest-index cycle node is not the smallest candidate overall: 0 acyclic, 3 == 5 cap 7, 5 == 3 cap 8 -> 0 and 5 accepted, 3 rejected."""
    # Candidates: 0 == 1 cap 2 (acyclic), 3 == 5 cap 7, 5 == 3 cap 8 (7, 8 primitive)
    gci0 = th.tensor(
        [
            [0, 1],
            [0, 2],
            [3, 5],
            [3, 7],
            [5, 3],
            [5, 8],
        ],
        dtype=th.long,
    )
    gci1 = th.tensor(
        [
            [1, 2, 0],
            [5, 7, 3],
            [3, 8, 5],
        ],
        dtype=th.long,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=9,
        k=2,
        max_components=16,
        top_idx=None,
        bottom_idx=None,
        enabled=True,
    )
    assert table.n_defined == 2
    assert table.def_mask.tolist() == [
        True,   # 0 accepted
        False,
        False,
        False,  # 3 rejected (breaks cycle)
        False,
        True,   # 5 accepted (with 3 primitive)
        False,
        False,
        False,
    ]
    assert table.def_x[0].item() == 1 and table.def_y[0].item() == 2
    assert table.def_x[5].item() == 3 and table.def_y[5].item() == 8
    assert table.level[0].item() == 1
    assert table.level[3].item() == 0
    assert table.level[5].item() == 1
    assert table.comps[5].item() == 4


def test_existence_targets_cm_requires_index_to_iri() -> None:
    """Passing cm without index_to_iri raises ValueError."""
    ds = PathDataset(fixture_path("t3.owl"))
    cm = CanonicalModel.from_dataset(ds)
    gci0_bot = th.zeros((0, 2), dtype=th.long)
    gci1 = th.zeros((0, 3), dtype=th.long)
    gci1_bot = th.zeros((0, 2), dtype=th.long)

    with pytest.raises(ValueError, match="index_to_iri"):
        ExistenceTargets.from_normal_forms(
            gci0_bot=gci0_bot,
            gci1=gci1,
            gci1_bot=gci1_bot,
            n_classes=5,
            top_idx=None,
            bottom_idx=None,
            index_to_iri=None,
            cm=cm,
        )


def test_existence_targets_asserted_drops_pairs_with_gci0_bot_class() -> None:
    """Under the asserted rule (no cm), a pair (C, D) where C or D has a gci0_bot row is dropped."""
    # Classes: 0:A, 1:B (in gci0_bot), 2:C, 3:BOT
    gci0_bot = th.tensor([[1, 3]], dtype=th.long)
    gci1 = th.tensor(
        [
            [0, 1, 2],  # involves 1 (unsat via gci0_bot) -> must be dropped
            [1, 2, 0],  # involves 1 (unsat via gci0_bot) -> must be dropped
            [0, 2, 1],  # both 0 and 2 satisfiable -> kept
        ],
        dtype=th.long,
    )
    gci1_bot = th.zeros((0, 2), dtype=th.long)

    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=gci0_bot,
        gci1=gci1,
        gci1_bot=gci1_bot,
        n_classes=4,
        top_idx=None,
        bottom_idx=3,
    )
    assert targets.classes.tolist() == [0, 2]
    assert targets.pairs.tolist() == [[0, 2]]


def test_gci0_and_gci1_raise_when_target_has_no_active_slots() -> None:
    """gci0_structural and gci1_structural raise ValueError when the target has no active slot in a row."""
    c = Regions(
        cen=th.tensor([[[0.0, 0.0]]], dtype=th.float64),
        w=th.tensor([[[1.0, 1.0]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    d_active = Regions(
        cen=th.tensor([[[0.0, 0.0]]], dtype=th.float64),
        w=th.tensor([[[1.0, 1.0]]], dtype=th.float64),
        active=th.tensor([[True]]),
    )
    target_inactive = Regions(
        cen=th.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=th.float64),
        w=th.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=th.float64),
        active=th.tensor([[False, False]]),
    )

    with pytest.raises(ValueError, match="active"):
        gci0_structural(c, target_inactive, margin=0.0)

    with pytest.raises(ValueError, match="active"):
        gci1_structural(c, d_active, target_inactive, margin=0.0)


def test_gci_structural_finite_grad_at_zero_violation() -> None:
    """Squared inclusion inside gci0_structural and gci1_structural has finite (zero) gradient at exact boundary violation == 0."""
    cen_c = th.tensor([[[1.0, 1.0]]], dtype=th.float64, requires_grad=True)
    w_c = th.tensor([[[0.5, 0.5]]], dtype=th.float64, requires_grad=True)
    cen_d = th.tensor([[[1.0, 1.0]]], dtype=th.float64, requires_grad=True)
    w_d = th.tensor([[[0.5, 0.5]]], dtype=th.float64, requires_grad=True)

    rc = Regions(cen=cen_c, w=w_c, active=th.tensor([[True]]))
    rd = Regions(cen=cen_d, w=w_d, active=th.tensor([[True]]))

    loss0 = gci0_structural(rc, rd, margin=0.0)
    loss1 = gci1_structural(rc, rd, rd, margin=0.0)
    (loss0 + loss1).backward()

    for t in (cen_c, w_c, cen_d, w_d):
        assert t.grad is not None
        assert bool(th.isfinite(t.grad).all())
        assert bool((t.grad == 0.0).all())


def test_gci1_bot_structural_agg_sum_and_invalid() -> None:
    """gci1_bot_structural(agg='sum') equals mean * n_active_pairs on a random batch with padding slots; invalid agg raises ValueError."""
    gen = th.Generator().manual_seed(20261102)
    b, kc, kd, dim = 4, 4, 5, 3
    cen_c = th.randn((b, kc, dim), generator=gen, dtype=th.float64) * 0.2
    w_c = th.abs(th.randn((b, kc, dim), generator=gen, dtype=th.float64)) + 0.5
    cen_d = th.randn((b, kd, dim), generator=gen, dtype=th.float64) * 0.2
    w_d = th.abs(th.randn((b, kd, dim), generator=gen, dtype=th.float64)) + 0.5

    # Varying active counts per row with padding slots (and put giant overlaps in inactive slots)
    act_c = th.tensor(
        [
            [True, True, False, False],
            [True, False, True, True],
            [False, True, False, False],
            [True, True, True, True],
        ],
        dtype=th.bool,
    )
    act_d = th.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, False, True, True, False],
            [False, True, False, True, False],
        ],
        dtype=th.bool,
    )
    w_c = th.where(act_c.unsqueeze(-1), w_c, th.full_like(w_c, 100.0))
    w_d = th.where(act_d.unsqueeze(-1), w_d, th.full_like(w_d, 100.0))

    rc = Regions(cen=cen_c, w=w_c, active=act_c)
    rd = Regions(cen=cen_d, w=w_d, active=act_d)

    loss_default = gci1_bot_structural(rc, rd, margin=0.05)
    loss_mean = gci1_bot_structural(rc, rd, margin=0.05, agg="mean")
    loss_sum = gci1_bot_structural(rc, rd, margin=0.05, agg="sum")

    assert th.equal(loss_default, loss_mean)
    n_active_pairs = (act_c.sum(dim=1) * act_d.sum(dim=1)).to(dtype=th.float64)
    assert n_active_pairs.tolist() == [6.0, 6.0, 3.0, 8.0]
    assert bool((loss_mean > 0.0).all())
    assert th.allclose(loss_sum, loss_mean * n_active_pairs, atol=1e-12, rtol=1e-12)

    for bad_agg in ("max", "min", "", 123, None):
        with pytest.raises(ValueError, match="agg"):
            gci1_bot_structural(rc, rd, margin=0.05, agg=bad_agg)  # type: ignore[arg-type]


def test_witness_targets_t3_with_and_without_cm(
    t3_dataset: PathDataset, t3_cm: CanonicalModel
) -> None:
    """On T3 without cm: pos(A)={(A,A),(A,X1),(A,X2)}, neg(A)={(A,B),(A,C)}; with cm: neg(A) also has (A,X3); never TOP/BOT."""
    from mowl.models import MultiBoxEL
    from mowl.nn.el.structural import build_structural_tables

    model_no_cm = MultiBoxEL(
        t3_dataset, embed_dim=2, objective="structural", canonical_model=None
    )
    _, targets_no_cm, top_idx = build_structural_tables(model_no_cm, cm=None)
    assert targets_no_cm.witness is not None
    wit_no_cm = targets_no_cm.witness

    cid = model_no_cm.class_index_dict
    a_idx = cid[NS + "A"]
    b_idx = cid[NS + "B"]
    c_idx = cid[NS + "C"]
    x1_idx = cid[NS + "X1"]
    x2_idx = cid[NS + "X2"]
    x3_idx = cid[NS + "X3"]
    bot_idx = cid.get("http://www.w3.org/2002/07/owl#Nothing")

    # Never include TOP or BOTTOM in classes, pos, or neg
    for wit in (wit_no_cm,):
        cls_set = set(wit.classes.tolist())
        pos_cls_set = set(wit.pos[:, 1].tolist())
        neg_cls_set = set(wit.neg[:, 1].tolist())
        for forbidden in (top_idx, bot_idx):
            if forbidden is not None:
                assert forbidden not in cls_set
                assert forbidden not in pos_cls_set
                assert forbidden not in neg_cls_set

    w_a_row = wit_no_cm.classes.tolist().index(a_idx)
    pos_a_no_cm = {
        int(cls_i)
        for r_i, cls_i in wit_no_cm.pos.tolist()
        if int(r_i) == w_a_row
    }
    neg_a_no_cm = {
        int(cls_i)
        for r_i, cls_i in wit_no_cm.neg.tolist()
        if int(r_i) == w_a_row
    }
    assert pos_a_no_cm == {a_idx, x1_idx, x2_idx}
    assert neg_a_no_cm == {b_idx, c_idx}
    assert x3_idx not in neg_a_no_cm

    # With T3 canonical model: neg(A) also includes X3 (since A cap X3 <= bot is entailed)
    _, targets_cm, _ = build_structural_tables(model_no_cm, cm=t3_cm)
    assert targets_cm.witness is not None
    wit_cm = targets_cm.witness

    for forbidden in (top_idx, bot_idx):
        if forbidden is not None:
            assert forbidden not in set(wit_cm.classes.tolist())
            assert forbidden not in set(wit_cm.pos[:, 1].tolist())
            assert forbidden not in set(wit_cm.neg[:, 1].tolist())

    w_a_row_cm = wit_cm.classes.tolist().index(a_idx)
    pos_a_cm = {
        int(cls_i)
        for r_i, cls_i in wit_cm.pos.tolist()
        if int(r_i) == w_a_row_cm
    }
    neg_a_cm = {
        int(cls_i)
        for r_i, cls_i in wit_cm.neg.tolist()
        if int(r_i) == w_a_row_cm
    }
    assert pos_a_cm == {a_idx, x1_idx, x2_idx}
    assert neg_a_cm == {b_idx, c_idx, x3_idx}

    # Rows are deduplicated and pos/neg are disjoint
    pos_tuples = [tuple(r) for r in wit_cm.pos.tolist()]
    neg_tuples = [tuple(r) for r in wit_cm.neg.tolist()]
    assert len(pos_tuples) == len(set(pos_tuples))
    assert len(neg_tuples) == len(set(neg_tuples))
    assert set(pos_tuples).isdisjoint(set(neg_tuples))

    # Passing cm without index_to_iri raises ValueError
    with pytest.raises(ValueError, match="index_to_iri"):
        WitnessTargets.from_normal_forms(
            classes=th.tensor([0], dtype=th.long),
            gci0=th.zeros((0, 2), dtype=th.long),
            gci1_bot=th.zeros((0, 2), dtype=th.long),
            cm=t3_cm,
        )


def test_inside_measure_and_witness_loss_zero_iff_and_inactive() -> None:
    """Hand-built regions: inside/outside/inactive behaviour of inside_measure and zero iff conditions of witness_loss."""
    # 3 rows, K=2 slots, d=2:
    # Row 0: slot 0 active box [-1, 1]^2 (cen=[0,0], w=[1,1]), slot 1 inactive giant box [-100,100]^2
    #        point p0 = [0.2, -0.5] -> min(1 - 0.2, 1 - 0.5) = +0.5 (strictly inside by 0.5)
    # Row 1: slot 0 active box [0, 2]^2 (cen=[1,1], w=[1,1]), slot 1 active box [10, 12]^2 (cen=[11,11], w=[1,1])
    #        point p1 = [2.4, 1.0] -> slot 0: min(1 - 1.4, 1 - 0) = -0.4; slot 1: min(1 - 8.6, 1 - 10) = -9.0 -> max = -0.4 (outside by 0.4)
    # Row 2: both slots inactive -> returns -1e6 with finite (zero) gradient
    pts = th.tensor(
        [[0.2, -0.5], [2.4, 1.0], [0.0, 0.0]],
        dtype=th.float64,
        requires_grad=True,
    )
    cen = th.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [11.0, 11.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=th.float64,
        requires_grad=True,
    )
    w = th.tensor(
        [
            [[1.0, 1.0], [100.0, 100.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ],
        dtype=th.float64,
        requires_grad=True,
    )
    active = th.tensor(
        [[True, False], [True, True], [False, False]], dtype=th.bool
    )
    reg = Regions(cen=cen, w=w, active=active)
    ins = inside_measure(pts, reg)
    assert ins.shape == (3,)
    assert ins[0].item() == pytest.approx(0.5, abs=1e-12)
    assert ins[1].item() == pytest.approx(-0.4, abs=1e-12)
    assert ins[2].item() == pytest.approx(-1e6, abs=1e-6)

    # Gradients on all-inactive row 2 are finite and zero, while rows 0 and 1 have finite non-zero gradients
    ins.sum().backward()
    for t in (pts, cen, w):
        assert t.grad is not None
        assert bool(th.isfinite(t.grad).all())
    assert bool((pts.grad[2] == 0.0).all())
    assert bool((cen.grad[2] == 0.0).all())
    assert bool((w.grad[2] == 0.0).all())
    assert bool((w.grad[0, 1] == 0.0).all())  # inactive slot 1 of row 0 gets 0 grad

    # witness_loss zero iff inside_pos >= m and inside_neg <= -m:
    # At margin m = 0.3, ins[0] = +0.5 >= 0.3 and ins[1] = -0.4 <= -0.3 -> loss == 0.0
    loss_zero = witness_loss(ins[:1].detach(), ins[1:2].detach(), margin=0.3)
    assert loss_zero.item() == 0.0

    # At boundary inside_pos == m and inside_neg == -m -> loss == 0.0
    loss_boundary = witness_loss(
        th.tensor([0.3], dtype=th.float64),
        th.tensor([-0.3], dtype=th.float64),
        margin=0.3,
    )
    assert loss_boundary.item() == 0.0

    # Violating positive (0.5 < m=0.6 -> (0.6 - 0.5)^2 = 0.01)
    loss_pos_viol = witness_loss(
        ins[:1].detach(), th.tensor([-1.0], dtype=th.float64), margin=0.6
    )
    assert loss_pos_viol.item() == pytest.approx(0.01, abs=1e-12)

    # Violating negative (-0.4 > -m=-0.6 -> (-0.4 + 0.6)^2 = 0.04)
    loss_neg_viol = witness_loss(
        th.tensor([1.0], dtype=th.float64), ins[1:2].detach(), margin=0.6
    )
    assert loss_neg_viol.item() == pytest.approx(0.04, abs=1e-12)

    # Empty tensors contribute 0.0
    empty = th.empty(0, dtype=th.float64)
    assert witness_loss(empty, empty, margin=0.1).item() == 0.0
    assert witness_loss(th.tensor([0.5], dtype=th.float64), empty, margin=0.1).item() == 0.0
    assert witness_loss(empty, th.tensor([-0.5], dtype=th.float64), margin=0.1).item() == 0.0


def test_slot_components_matches_pair_slots_two_levels() -> None:
    """On a two-level definition at k=2, every materialised slot box equals the intersection of the primitive component boxes mapped by slot_components."""
    from mowl.nn import MultiBoxELModule

    # Classes:
    # 0: A := Aux cap X3 (level 2, 8 active slots, def_x=1 (level 1), def_y=4 (prim))
    # 1: Aux := X1 cap X2 (level 1, 4 active slots, def_x=2, def_y=3)
    # 2: X1 (primitive, 2 slots)
    # 3: X2 (primitive, 2 slots)
    # 4: X3 (primitive, 2 slots)
    # 5: A_rev := X1 cap Aux2 (level 2, 8 active slots, def_x=2 (prim), def_y=6 (level 1))
    # 6: Aux2 := X2 cap X3 (level 1, 4 active slots, def_x=3, def_y=4)
    gci0 = th.tensor(
        [
            [1, 2],  # Aux <= X1
            [1, 3],  # Aux <= X2
            [0, 1],  # A <= Aux
            [0, 4],  # A <= X3
            [6, 3],  # Aux2 <= X2
            [6, 4],  # Aux2 <= X3
            [5, 2],  # A_rev <= X1
            [5, 6],  # A_rev <= Aux2
        ],
        dtype=th.long,
    )
    gci1 = th.tensor(
        [
            [2, 3, 1],  # X1 cap X2 <= Aux
            [1, 4, 0],  # Aux cap X3 <= A
            [3, 4, 6],  # X2 cap X3 <= Aux2
            [2, 6, 5],  # X1 cap Aux2 <= A_rev
        ],
        dtype=th.long,
    )
    table = DefinitionTable.from_normal_forms(
        gci0=gci0,
        gci1=gci1,
        n_classes=7,
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
        n_classes=7,
        top_idx=None,
        bottom_idx=None,
    )

    th.manual_seed(20261102)
    mod = MultiBoxELModule(
        nb_ont_classes=7,
        nb_rels=1,
        embed_dim=3,
        n_components=2,
        objective="structural",
        max_components=16,
    )
    mod.set_structural_tables(table, targets)

    # Check primitive classes (2, 3, 4)
    for prim_cls in (2, 3, 4):
        for s in range(2):
            assert slot_components(table, prim_cls, s, 2) == {prim_cls: s}
            assert slot_components(table, prim_cls, s) == {prim_cls: s}
            assert slot_components(mod, prim_cls, s) == {prim_cls: s}

    raw_cen, raw_off = mod.class_boxes(th.arange(7, dtype=th.long))
    all_reg = mod.class_regions(th.arange(7, dtype=th.long))

    # Check level-1 (Aux=1, Aux2=6) and level-2 (A=0, A_rev=5) defined classes against pair_slots / class_regions
    for def_cls, n_slots, expected_prims in (
        (1, 4, {2, 3}),
        (6, 4, {3, 4}),
        (0, 8, {2, 3, 4}),
        (5, 8, {2, 3, 4}),
    ):
        assert int(all_reg.active[def_cls].sum().item()) == n_slots
        seen_maps = []
        for s in range(n_slots):
            comp_map = slot_components(table, def_cls, s, 2)
            assert comp_map == slot_components(mod, def_cls, s)
            assert set(comp_map.keys()) == expected_prims
            seen_maps.append(tuple(sorted(comp_map.items())))

            # Intersect the mapped primitive component boxes directly
            lowers = [
                raw_cen[p_idx, c_idx] - raw_off[p_idx, c_idx]
                for p_idx, c_idx in comp_map.items()
            ]
            uppers = [
                raw_cen[p_idx, c_idx] + raw_off[p_idx, c_idx]
                for p_idx, c_idx in comp_map.items()
            ]
            lo = th.stack(lowers, dim=0).amax(dim=0)
            hi = th.stack(uppers, dim=0).amin(dim=0)
            expected_cen = (lo + hi) / 2.0
            expected_w = (hi - lo) / 2.0

            assert th.allclose(all_reg.cen[def_cls, s], expected_cen, atol=1e-6)
            assert th.allclose(all_reg.w[def_cls, s], expected_w, atol=1e-6)

        # All n_slots active slots correspond to distinct primitive component combinations
        assert len(set(seen_maps)) == n_slots

    # Out-of-bounds and invalid arguments raise ValueError
    with pytest.raises(ValueError, match="slot"):
        slot_components(table, 2, 2, 2)
    with pytest.raises(ValueError, match="slot"):
        slot_components(table, 0, 8, 2)
    with pytest.raises(ValueError, match="slot"):
        slot_components(table, 0, -1, 2)
    with pytest.raises(ValueError, match="cls"):
        slot_components(table, 7, 0, 2)
    with pytest.raises(ValueError, match="cls"):
        slot_components(table, -1, 0, 2)
    with pytest.raises(ValueError, match="k"):
        slot_components(table, 0, 0, 0)
    with pytest.raises(ValueError, match="k"):
        slot_components(table, 0, 0, True)  # type: ignore[arg-type]





