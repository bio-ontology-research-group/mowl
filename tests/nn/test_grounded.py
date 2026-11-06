"""Unit tests for mowl.nn.el.grounded (losses, GroundedELModule, sampling, R1/R2/R3)."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch as th
import torch.nn as nn

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.datasets.el import ELDataset
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.nn import GroundedELModule
from mowl.nn.el.grounded import (
    RegionSample,
    adversarial_ascent,
    class_assertion,
    component_extremes,
    crisp_membership,
    fact,
    gci0,
    gci0_bot,
    gci0_bot_region,
    gci0_region,
    gci1,
    gci1_bot,
    gci1_bot_region,
    gci1_region,
    gci2,
    gci2_dead,
    gci2_region,
    gci3,
    gci3_bot,
    gci3_bot_region,
    gci3_region,
    minimality,
    role_assertion,
    role_chain,
    role_chain_region,
    role_inclusion,
    role_inclusion_region,
    sample_active_component,
    sample_class_region,
    sample_component_intersection,
    sample_elements,
    sample_filler_elements,
    sample_inside_elements,
    sample_pair_region,
    sample_role_region,
)
from mowl.nn.el.structural import Regions
from mowl.owlapi import OWLAPIAdapter
from tests.geometric.helpers import (
    IND,
    NS,
    fixture_path,
    roles_cm,
    roles_dataset,
    t3_cm,
    t3_dataset,
)


# ---------------------------------------------------------------------------
# 1. Pure loss function tests (§3 table & §4 region reduction)
# ---------------------------------------------------------------------------


def test_every_hinge_zero_on_crisp_model_and_positive_on_violation() -> None:
    """Every §3 hinge is exact 0 on a crisp satisfying assignment and > 0 on a violating one."""
    one = th.tensor([1.0])
    zero = th.tensor([0.0])

    # fact: live_x = 1 - delta_x <= m_A(x_A)
    assert fact(one, dead=zero).item() == 0.0
    assert fact(zero, dead=one).item() == 0.0
    assert fact(one, dead=None).item() == 0.0
    assert fact(zero, dead=zero).item() == pytest.approx(1.0)
    assert fact(zero, dead=None).item() == pytest.approx(1.0)

    # gci0: A <= B
    for ma, mb in ((1.0, 1.0), (0.0, 1.0), (0.0, 0.0)):
        assert gci0(th.tensor([ma]), th.tensor([mb]), dead=zero).item() == 0.0
        assert gci0(th.tensor([ma]), th.tensor([mb]), dead=None).item() == 0.0
    assert gci0(one, zero, dead=zero).item() == pytest.approx(1.0)
    assert gci0(one, zero, dead=None).item() == pytest.approx(1.0)

    # gci1: A cap B <= C
    for ma, mb, mc in (
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ):
        assert (
            gci1(th.tensor([ma]), th.tensor([mb]), th.tensor([mc]), dead=zero).item()
            == 0.0
        )
        assert (
            gci1(th.tensor([ma]), th.tensor([mb]), th.tensor([mc]), dead=None).item()
            == 0.0
        )
    assert gci1(one, one, zero, dead=zero).item() == pytest.approx(1.0)
    assert gci1(one, one, zero, dead=None).item() == pytest.approx(1.0)

    # gci0_bot: A <= bot
    assert gci0_bot(zero, dead=zero).item() == 0.0
    assert gci0_bot(zero, dead=None).item() == 0.0
    assert gci0_bot(one, dead=zero).item() == pytest.approx(1.0)
    assert gci0_bot(one, dead=None).item() == pytest.approx(1.0)

    # gci1_bot: A cap B <= bot
    for ma, mb in ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)):
        assert gci1_bot(th.tensor([ma]), th.tensor([mb]), dead=zero).item() == 0.0
        assert gci1_bot(th.tensor([ma]), th.tensor([mb]), dead=None).item() == 0.0
    assert gci1_bot(one, one, dead=zero).item() == pytest.approx(1.0)
    assert gci1_bot(one, one, dead=None).item() == pytest.approx(1.0)

    # gci2: A <= exists r.B (including dead propagation)
    assert gci2(one, one, dead=zero, dead_filler=zero).item() == 0.0
    assert gci2(zero, zero, dead=zero, dead_filler=one).item() == 0.0
    assert gci2(one, one, dead=None, dead_filler=zero).item() == 0.0
    assert gci2(one, zero, dead=zero, dead_filler=zero).item() == pytest.approx(1.0)
    assert gci2(one, one, dead=zero, dead_filler=one).item() == pytest.approx(1.0)
    assert gci2(one, zero, dead=None, dead_filler=None).item() == pytest.approx(1.0)

    # gci3: exists r.B <= C
    for rho_v, mb, mc in (
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ):
        assert (
            gci3(th.tensor([rho_v]), th.tensor([mb]), th.tensor([mc]), dead=zero).item()
            == 0.0
        )
        assert (
            gci3(th.tensor([rho_v]), th.tensor([mb]), th.tensor([mc]), dead=None).item()
            == 0.0
        )
    assert gci3(one, one, zero, dead=zero).item() == pytest.approx(1.0)
    assert gci3(one, one, zero, dead=None).item() == pytest.approx(1.0)

    # gci3_bot: exists r.B <= bot
    for rho_v, mb in ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)):
        assert gci3_bot(th.tensor([rho_v]), th.tensor([mb]), dead=zero).item() == 0.0
        assert gci3_bot(th.tensor([rho_v]), th.tensor([mb]), dead=None).item() == 0.0
    assert gci3_bot(one, one, dead=zero).item() == pytest.approx(1.0)
    assert gci3_bot(one, one, dead=None).item() == pytest.approx(1.0)

    # role_inclusion: r <= s
    for rr, rs in ((1.0, 1.0), (0.0, 1.0), (0.0, 0.0)):
        assert role_inclusion(th.tensor([rr]), th.tensor([rs])).item() == 0.0
    assert role_inclusion(one, zero).item() == pytest.approx(1.0)

    # role_chain: r o s <= t
    for rr, rs, rt in (
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ):
        assert (
            role_chain(th.tensor([rr]), th.tensor([rs]), th.tensor([rt])).item() == 0.0
        )
    assert role_chain(one, one, zero).item() == pytest.approx(1.0)

    # ABox assertions: A(a), r(a, b)
    assert class_assertion(one).item() == 0.0
    assert class_assertion(zero).item() == pytest.approx(1.0)
    assert role_assertion(one).item() == 0.0
    assert role_assertion(zero).item() == pytest.approx(1.0)

    # minimality and crisp_membership
    depths = th.tensor([-0.5, 0.0, 0.25])
    assert crisp_membership(depths).tolist() == [0.0, 0.0, 1.0]
    min_loss = minimality(
        m_rand=th.tensor([0.5, 0.5]),
        rho_rand=th.tensor([0.2, 0.2]),
        dead=th.tensor([0.1, 0.1]),
        eps=0.1,
    )
    assert min_loss.item() == pytest.approx(0.1 * (0.5 + 0.2 + 0.1))
    assert minimality(one, one, one, eps=0.0).item() == 0.0


def test_hand_built_geometric_model_crisp_hinges() -> None:
    """Hand-built 2-D GroundedELModule satisfies all crisp hinges and fires on violations."""
    # Classes: 0=A, 1=B, 2=C, 3=D, 4=Unsat; Roles: 0=r, 1=s, 2=t; Individual: 5=ind_a
    # Theory: A <= B, A cap B <= C, Unsat <= bot, B cap D <= bot, A <= exists r.D,
    #         exists s.D <= C, exists t.A <= bot, r <= s, r o s <= t, A(ind_a), r(ind_a, x_D)
    mod = GroundedELModule(
        nb_elements=6,
        nb_classes=5,
        nb_rels=3,
        embed_dim=2,
        n_components=2,
        n_role_components=2,
        tau=0.1,
    )
    pts = th.tensor(
        [
            [0.0, 0.0],   # 0: x_A
            [2.0, 0.0],   # 1: x_B
            [4.0, 0.0],   # 2: x_C
            [6.0, 0.0],   # 3: x_D
            [8.0, 0.0],   # 4: x_Unsat (dead)
            [0.05, 0.0],  # 5: ind_a (inside A's box around [0, 0])
        ]
    )
    # Boxes:
    # A (0): box at [0, 0] hw=0.2 (contains x_A and ind_a)
    # B (1): box 0 at [0, 0] hw=0.3, box 1 at [2, 0] hw=0.2 (contains x_A, ind_a, x_B)
    # C (2): box 0 at [0, 0] hw=0.3, box 1 at [4, 0] hw=0.2, box 2 at [6, 0] hw=0.2 (contains x_A, ind_a, x_C, x_D)
    # D (3): box 0 at [6, 0] hw=0.2 (contains x_D; disjoint from B)
    # Unsat (4): inactive
    labels = th.tensor(
        [
            [1, 1, 1, 0, 0],  # x_A in A, B, C
            [0, 1, 0, 0, 0],  # x_B in B
            [0, 0, 1, 0, 0],  # x_C in C
            [0, 0, 1, 1, 0],  # x_D in C, D
            [0, 0, 0, 0, 0],  # x_Unsat dead
            [1, 1, 1, 0, 0],  # ind_a in A, B, C
        ],
        dtype=th.long,
    )
    edges = [
        (0, 3, 0),  # (x_A, x_D) in r
        (5, 3, 0),  # (ind_a, x_D) in r
        (0, 3, 1),  # (x_A, x_D) in s
        (5, 3, 1),  # (ind_a, x_D) in s
        (3, 3, 1),  # (x_D, x_D) in s (so r o s has (x_A, x_D) and (ind_a, x_D) in t)
        (0, 3, 2),  # (x_A, x_D) in t
        (5, 3, 2),  # (ind_a, x_D) in t
    ]
    mod.expand_components(6, new_role_K=3)
    mod.exact_limit_init(labels, edges, delta=0.1, seed=7)
    # Override pts to our exact hand coordinates to test manual geometry directly
    with th.no_grad():
        mod.elem_pos.weight.copy_(pts)
        # Re-centre active class/role boxes on pts
        for c_id in range(5):
            for k_id in range(6):
                if mod.class_active[c_id, k_id]:
                    mod.class_centers[c_id, k_id].copy_(pts[k_id])
        for r_id, r_edges in enumerate(
            [
                [(0, 3), (5, 3)],
                [(0, 3), (5, 3), (3, 3)],
                [(0, 3), (5, 3)],
            ]
        ):
            for k_id, (u, v) in enumerate(r_edges):
                mod.role_head_centers[r_id, k_id].copy_(pts[u])
                mod.role_tail_centers[r_id, k_id].copy_(pts[v])

    all_pos = mod.positions()
    dead_vec = crisp_membership(mod.dead() - 0.5)
    assert dead_vec.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    def cm_cls(c_idx: int) -> th.Tensor:
        return crisp_membership(mod.depth(all_pos, c_idx))

    m_A, m_B, m_C, m_D, m_U = (cm_cls(i) for i in range(5))

    # Fact for class elements 0..4
    cls_elem_ids = th.arange(5)
    self_m = crisp_membership(mod.depth(mod.positions(cls_elem_ids), cls_elem_ids))
    assert th.all(fact(self_m, dead=dead_vec[:5]) == 0.0)

    # GCI0: A <= B
    assert th.all(gci0(m_A, m_B, dead=dead_vec) == 0.0)
    # Violating GCI0: B <= A fails at x_B
    assert gci0(m_B, m_A, dead=dead_vec)[1].item() == 1.0

    # GCI1: A cap B <= C
    assert th.all(gci1(m_A, m_B, m_C, dead=dead_vec) == 0.0)
    # Violating GCI1: A cap B <= D fails at x_A
    assert gci1(m_A, m_B, m_D, dead=dead_vec)[0].item() == 1.0

    # GCI0_bot: Unsat <= bot
    assert th.all(gci0_bot(m_U, dead=dead_vec) == 0.0)
    assert gci0_bot(m_A, dead=dead_vec)[0].item() == 1.0

    # GCI1_bot: B cap D <= bot
    assert th.all(gci1_bot(m_B, m_D, dead=dead_vec) == 0.0)
    assert gci1_bot(m_A, m_B, dead=dead_vec)[0].item() == 1.0

    # GCI2: A <= exists r.D
    rho_x_xD_r = crisp_membership(mod.role_depth(all_pos, all_pos[3], 0))
    dead_D = dead_vec[3].expand_as(dead_vec)
    assert th.all(gci2(m_A, rho_x_xD_r, dead=dead_vec, dead_filler=dead_D) == 0.0)
    # Violating GCI2: B <= exists r.D fails at x_B
    assert gci2(m_B, rho_x_xD_r, dead=dead_vec, dead_filler=dead_D)[1].item() == 1.0

    # GCI3: exists s.D <= C
    for y_idx in range(6):
        rho_xy_s = crisp_membership(mod.role_depth(all_pos, all_pos[y_idx], 1))
        mb_y = m_D[y_idx].expand_as(rho_xy_s)
        assert th.all(gci3(rho_xy_s, mb_y, m_C, dead=dead_vec) == 0.0)

    # GCI3_bot: exists t.A <= bot
    for y_idx in range(6):
        rho_xy_t = crisp_membership(mod.role_depth(all_pos, all_pos[y_idx], 2))
        ma_y = m_A[y_idx].expand_as(rho_xy_t)
        assert th.all(gci3_bot(rho_xy_t, ma_y, dead=dead_vec) == 0.0)

    # ABox class & role assertions: A(ind_a), r(ind_a, x_D)
    assert class_assertion(m_A[5:6]).item() == 0.0
    assert class_assertion(m_D[5:6]).item() == 1.0
    assert role_assertion(rho_x_xD_r[5:6]).item() == 0.0


# ---------------------------------------------------------------------------
# 2. Dead element silencing and dead propagation tests
# ---------------------------------------------------------------------------


def test_dead_element_silences_hinges_and_propagates_across_gci2() -> None:
    """delta_x = 1 silences all element hinges at x; m_A = 0, delta_x = 0 leaves only fact > 0; gci2 propagates deadness."""
    one = th.tensor([1.0])
    zero = th.tensor([0.0])

    # At delta_x = 1, even maximally violating memberships (m_A=1, m_B=0, m_C=0, rho=1) give 0
    assert gci0(one, zero, dead=one).item() == 0.0
    assert gci1(one, one, zero, dead=one).item() == 0.0
    assert gci0_bot(one, dead=one).item() == 0.0
    assert gci1_bot(one, one, dead=one).item() == 0.0
    assert gci2(one, zero, dead=one, dead_filler=one).item() == 0.0
    assert gci3(one, one, zero, dead=one).item() == 0.0
    assert gci3_bot(one, one, dead=one).item() == 0.0
    assert fact(zero, dead=one).item() == 0.0

    # At m = 0 and delta_x = 0, all GCI element hinges are 0 EXCEPT fact (which is 1.0)
    # and ABox assertions (which have no delta and evaluate to 1.0)
    assert gci0(zero, zero, dead=zero).item() == 0.0
    assert gci1(zero, one, zero, dead=zero).item() == 0.0
    assert gci0_bot(zero, dead=zero).item() == 0.0
    assert gci1_bot(zero, one, dead=zero).item() == 0.0
    assert gci2(zero, zero, dead=zero, dead_filler=one).item() == 0.0
    assert gci3(zero, one, zero, dead=zero).item() == 0.0
    assert gci3_bot(zero, one, dead=zero).item() == 0.0
    assert fact(zero, dead=zero).item() == pytest.approx(1.0)
    assert fact(zero, dead=None).item() == pytest.approx(1.0)
    assert class_assertion(zero).item() == pytest.approx(1.0)
    assert role_assertion(zero).item() == pytest.approx(1.0)

    # Dead propagation in gci2 (A <= exists r.B):
    # When x is in A (m_A = 1), edge (x, x_B, r) is present (rho = 1),
    # and filler x_B is dead (delta_{x_B} = 1) while x is live (delta_x = 0):
    # edge term is relu(1 - 1 - 0) = 0, dead propagation term is relu(1 + 1 - 1 - 0) = 1.0
    assert gci2_dead(one, dead_filler=one, dead=zero).item() == pytest.approx(1.0)
    assert gci2(one, one, dead=zero, dead_filler=one).item() == pytest.approx(1.0)
    # Once delta_x rises to 1, dead propagation is satisfied (0.0)
    assert gci2_dead(one, dead_filler=one, dead=one).item() == 0.0
    assert gci2(one, one, dead=one, dead_filler=one).item() == 0.0


# ---------------------------------------------------------------------------
# 3. Monotone capacity (R3): expand_components bitwise equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg", ["max", "lse"])
def test_expand_components_bitwise_equal(agg: str) -> None:
    """expand_components leaves depth, membership, role_depth, and role_membership bitwise identical."""
    th.manual_seed(123)
    mod = GroundedELModule(
        nb_elements=8,
        nb_classes=6,
        nb_rels=3,
        embed_dim=4,
        n_components=2,
        n_role_components=2,
        tau=0.25,
        agg=agg,
        agg_tau=0.1,
        top_id=4,
        bot_id=5,
    )
    # Also mark one class slot and one role slot inactive to test mixed active/inactive before expansion
    with th.no_grad():
        mod.class_active[1, :] = False
        mod.role_active[2, :] = False

    gen = th.Generator().manual_seed(99)
    pts_x = th.randn((20, 4), generator=gen)
    pts_y = th.randn((20, 4), generator=gen)
    cls_ids = th.randint(0, 6, (20,), generator=gen)
    rel_ids = th.randint(0, 3, (20,), generator=gen)

    with th.no_grad():
        dep_before = mod.depth(pts_x, cls_ids).clone()
        mem_before = mod.membership(pts_x, cls_ids).clone()
        rdep_before = mod.role_depth(pts_x, pts_y, rel_ids).clone()
        rmem_before = mod.role_membership(pts_x, pts_y, rel_ids).clone()

    mod.expand_components(new_K=7, new_role_K=5)
    assert mod.n_components == 7
    assert mod.n_role_components == 5
    assert mod.class_centers.shape == (6, 7, 4)
    assert mod.role_head_centers.shape == (3, 5, 4)

    with th.no_grad():
        dep_after = mod.depth(pts_x, cls_ids)
        mem_after = mod.membership(pts_x, cls_ids)
        rdep_after = mod.role_depth(pts_x, pts_y, rel_ids)
        rmem_after = mod.role_membership(pts_x, pts_y, rel_ids)

    assert th.equal(dep_before, dep_after)
    assert th.equal(mem_before, mem_after)
    assert th.equal(rdep_before, rdep_after)
    assert th.equal(rmem_before, rmem_after)

    with pytest.raises(ValueError, match="new_K"):
        mod.expand_components(new_K=3)


# ---------------------------------------------------------------------------
# 4. Exact-limit construction (R2, §5) on T3 and role theories
# ---------------------------------------------------------------------------


def _build_least_model_tables_from_cm(
    el_ds: ELDataset,
    cm: CanonicalModel,
) -> Tuple[th.Tensor, List[Tuple[int, int, int]], Optional[int], Optional[int], th.Tensor]:
    """Compute (labels, edges, top_id, bot_id, sat_mask) in the test from ELDataset and CanonicalModel."""
    class_to_idx: Dict[str, int] = dict(el_ds.class_index_dict)
    rel_to_idx: Dict[str, int] = dict(el_ds.object_property_index_dict)
    ind_to_idx: Dict[str, int] = (
        dict(el_ds._individual_index_dict) if el_ds._individual_index_dict else {}
    )

    nb_classes = len(class_to_idx)
    nb_inds = len(ind_to_idx)
    nb_elements = nb_classes + nb_inds

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    idx_to_ind = {v: k for k, v in ind_to_idx.items()}

    top_id = class_to_idx.get(TOP, None)
    bot_id = class_to_idx.get(BOTTOM, None)

    labels = th.zeros((nb_elements, nb_classes), dtype=th.bool)
    sat_classes = th.zeros((nb_classes,), dtype=th.bool)

    # Map each live element k in 0 .. nb_elements - 1 to its CanonicalModel position
    elem_to_cm_pos: Dict[int, int] = {}
    for c_idx, c_iri in idx_to_class.items():
        if c_iri == BOTTOM:
            continue
        if cm.is_satisfiable(c_iri):
            sat_classes[c_idx] = True
            elem_to_cm_pos[c_idx] = cm.element_of(c_iri)

    for i_idx, i_iri in idx_to_ind.items():
        elem_idx = nb_classes + i_idx
        elem_to_cm_pos[elem_idx] = cm.element_of(i_iri)

    for elem_idx, cm_pos in elem_to_cm_pos.items():
        for a_idx, a_iri in idx_to_class.items():
            if a_iri == TOP:
                labels[elem_idx, a_idx] = True
            elif a_iri == BOTTOM:
                labels[elem_idx, a_idx] = False
            elif cm_pos in cm.extension(a_iri):
                labels[elem_idx, a_idx] = True

    cm_pos_to_elems: Dict[int, List[int]] = {}
    for elem_idx, cm_pos in elem_to_cm_pos.items():
        cm_pos_to_elems.setdefault(cm_pos, []).append(elem_idx)

    edges: List[Tuple[int, int, int]] = []
    for r_iri, r_idx in rel_to_idx.items():
        for src_pos, dst_pos in cm.role(r_iri):
            for x_el in cm_pos_to_elems.get(src_pos, ()):
                for y_el in cm_pos_to_elems.get(dst_pos, ()):
                    edges.append((x_el, y_el, r_idx))

    return labels, edges, top_id, bot_id, sat_classes


def _assert_exact_limit_zero_hinges(
    el_ds: ELDataset,
    cm: CanonicalModel,
    n_region_samples: int = 2000,
    seed: int = 42,
) -> None:
    """Initialise GroundedELModule via exact_limit_init and assert all crisp element & region hinges == 0."""
    labels, edges, top_id, bot_id, sat_classes = _build_least_model_tables_from_cm(
        el_ds, cm
    )
    nb_classes = labels.shape[1]
    nb_elements = labels.shape[0]
    nb_rels = len(el_ds.object_property_index_dict)

    mod = GroundedELModule(
        nb_elements=nb_elements,
        nb_classes=nb_classes,
        nb_rels=nb_rels,
        embed_dim=3,
        n_components=nb_elements,
        n_role_components=max(1, len(edges)),
        tau=0.05,
        top_id=top_id,
        bot_id=bot_id,
    )
    mod.exact_limit_init(labels, edges, delta=0.05, seed=seed)

    # Check live elements == satisfiable classes (and live individuals)
    dead_crisp = (mod.dead() > 0.5)
    live_cls_elements = ~dead_crisp[:nb_classes]
    assert th.equal(live_cls_elements, sat_classes), (
        f"Live class elements {live_cls_elements.tolist()} != satisfiable classes {sat_classes.tolist()}"
    )
    if nb_elements > nb_classes:
        assert not bool(dead_crisp[nb_classes:].any().item())

    all_pos = mod.positions()  # (nb_elements, d)
    dead_f = dead_crisp.to(dtype=th.float32)

    # Precompute crisp element membership matrix M[x, A] of shape (nb_elements, nb_classes)
    elem_m = th.stack(
        [
            crisp_membership(mod.depth(all_pos, c_idx))
            for c_idx in range(nb_classes)
        ],
        dim=1,
    )

    # 1. Fact hinge for all class elements (except bot_id)
    for c_idx in range(nb_classes):
        if c_idx == bot_id:
            continue
        f_val = fact(elem_m[c_idx : c_idx + 1, c_idx], dead=dead_f[c_idx : c_idx + 1])
        assert f_val.item() == 0.0, f"Fact hinge non-zero for class {c_idx}: {f_val.item()}"

    gci_datasets = el_ds.get_gci_datasets()
    gen = th.Generator().manual_seed(seed + 100)

    # 2. GCI0: A <= B (elements + 2000 region samples)
    if "gci0" in gci_datasets and len(gci_datasets["gci0"]) > 0:
        axioms = gci_datasets["gci0"][:]
        for row in axioms:
            a_id, b_id = int(row[0].item()), int(row[1].item())
            h_elem = gci0(elem_m[:, a_id], elem_m[:, b_id], dead=dead_f)
            assert float(h_elem.max().item()) == 0.0
            reg = sample_class_region(mod, a_id, generator=gen, n_samples=n_region_samples)
            if bool(reg.valid.any().item()):
                pts = reg.points[reg.valid]
                ma = crisp_membership(mod.depth(pts, a_id))
                mb = crisp_membership(mod.depth(pts, b_id))
                h_reg = gci0(ma, mb, dead=None)
                assert float(h_reg.max().item()) == 0.0

    # 3. GCI0_bot: A <= bot (elements + 2000 region samples)
    if "gci0_bot" in gci_datasets and len(gci_datasets["gci0_bot"]) > 0:
        axioms = gci_datasets["gci0_bot"][:]
        for row in axioms:
            a_id = int(row[0].item())
            h_elem = gci0_bot(elem_m[:, a_id], dead=dead_f)
            assert float(h_elem.max().item()) == 0.0
            reg = sample_class_region(mod, a_id, generator=gen, n_samples=n_region_samples)
            assert not bool(reg.valid.any().item()), f"Unsatisfiable class {a_id} had active region"

    # 4. GCI1: A cap B <= C (elements + 2000 region samples)
    if "gci1" in gci_datasets and len(gci_datasets["gci1"]) > 0:
        axioms = gci_datasets["gci1"][:]
        for row in axioms:
            a_id, b_id, c_id = int(row[0].item()), int(row[1].item()), int(row[2].item())
            h_elem = gci1(elem_m[:, a_id], elem_m[:, b_id], elem_m[:, c_id], dead=dead_f)
            assert float(h_elem.max().item()) == 0.0
            reg = sample_pair_region(mod, a_id, b_id, generator=gen, n_samples=n_region_samples)
            if bool(reg.valid.any().item()):
                pts = reg.points[reg.valid]
                ma = crisp_membership(mod.depth(pts, a_id))
                mb = crisp_membership(mod.depth(pts, b_id))
                mc = crisp_membership(mod.depth(pts, c_id))
                h_reg = gci1(ma, mb, mc, dead=None)
                assert float(h_reg.max().item()) == 0.0

    # 5. GCI1_bot: A cap B <= bot (elements + 2000 region samples)
    if "gci1_bot" in gci_datasets and len(gci_datasets["gci1_bot"]) > 0:
        axioms = gci_datasets["gci1_bot"][:]
        for row in axioms:
            a_id, b_id = int(row[0].item()), int(row[1].item())
            h_elem = gci1_bot(elem_m[:, a_id], elem_m[:, b_id], dead=dead_f)
            assert float(h_elem.max().item()) == 0.0
            # Sample both from intersection and from A's region directly
            reg_inter = sample_pair_region(mod, a_id, b_id, generator=gen, n_samples=n_region_samples)
            assert not bool(reg_inter.valid.any().item())
            reg_a = sample_class_region(mod, a_id, generator=gen, n_samples=n_region_samples)
            if bool(reg_a.valid.any().item()):
                pts = reg_a.points[reg_a.valid]
                ma = crisp_membership(mod.depth(pts, a_id))
                mb = crisp_membership(mod.depth(pts, b_id))
                assert float(gci1_bot(ma, mb, dead=None).max().item()) == 0.0

    # 6. GCI2: A <= exists r.B (elements + 2000 region samples)
    if "gci2" in gci_datasets and len(gci_datasets["gci2"]) > 0:
        axioms = gci_datasets["gci2"][:]
        for row in axioms:
            a_id, r_id, b_id = int(row[0].item()), int(row[1].item()), int(row[2].item())
            rho_x_xb = crisp_membership(mod.role_depth(all_pos, all_pos[b_id], r_id))
            dead_b = dead_f[b_id].expand_as(dead_f)
            h_elem = gci2(elem_m[:, a_id], rho_x_xb, dead=dead_f, dead_filler=dead_b)
            assert float(h_elem.max().item()) == 0.0
            reg = sample_class_region(mod, a_id, generator=gen, n_samples=n_region_samples)
            if bool(reg.valid.any().item()):
                pts = reg.points[reg.valid]
                ma = crisp_membership(mod.depth(pts, a_id))
                rho_z = crisp_membership(mod.role_depth(pts, all_pos[b_id], r_id))
                db_z = dead_f[b_id].expand_as(ma)
                h_reg = gci2(ma, rho_z, dead=None, dead_filler=db_z)
                assert float(h_reg.max().item()) == 0.0

    # 7. GCI3: exists r.B <= C (all element pairs + 2000 region samples)
    if "gci3" in gci_datasets and len(gci_datasets["gci3"]) > 0:
        axioms = gci_datasets["gci3"][:]
        for row in axioms:
            r_id, b_id, c_id = int(row[0].item()), int(row[1].item()), int(row[2].item())
            for y_id in range(nb_elements):
                rho_xy = crisp_membership(mod.role_depth(all_pos, all_pos[y_id], r_id))
                mb_y = elem_m[y_id, b_id].expand_as(rho_xy)
                h_elem = gci3(rho_xy, mb_y, elem_m[:, c_id], dead=dead_f)
                assert float(h_elem.max().item()) == 0.0
            h_samp, t_samp = sample_role_region(mod, r_id, generator=gen, n_samples=n_region_samples)
            if bool(h_samp.valid.any().item()):
                zx = h_samp.points[h_samp.valid]
                zy = t_samp.points[t_samp.valid]
                rho_z = crisp_membership(mod.role_depth(zx, zy, r_id))
                mb_z = crisp_membership(mod.depth(zy, b_id))
                mc_z = crisp_membership(mod.depth(zx, c_id))
                h_reg = gci3(rho_z, mb_z, mc_z, dead=None)
                assert float(h_reg.max().item()) == 0.0

    # 8. GCI3_bot: exists r.B <= bot (all element pairs + 2000 region samples)
    if "gci3_bot" in gci_datasets and len(gci_datasets["gci3_bot"]) > 0:
        axioms = gci_datasets["gci3_bot"][:]
        for row in axioms:
            r_id, b_id = int(row[0].item()), int(row[1].item())
            for y_id in range(nb_elements):
                rho_xy = crisp_membership(mod.role_depth(all_pos, all_pos[y_id], r_id))
                mb_y = elem_m[y_id, b_id].expand_as(rho_xy)
                h_elem = gci3_bot(rho_xy, mb_y, dead=dead_f)
                assert float(h_elem.max().item()) == 0.0
            h_samp, t_samp = sample_role_region(mod, r_id, generator=gen, n_samples=n_region_samples)
            if bool(h_samp.valid.any().item()):
                zx = h_samp.points[h_samp.valid]
                zy = t_samp.points[t_samp.valid]
                rho_z = crisp_membership(mod.role_depth(zx, zy, r_id))
                mb_z = crisp_membership(mod.depth(zy, b_id))
                h_reg = gci3_bot(rho_z, mb_z, dead=None)
                assert float(h_reg.max().item()) == 0.0

    # 9. Role inclusion: r <= s (all element pairs + 2000 region samples)
    if "role_inclusion" in gci_datasets and len(gci_datasets["role_inclusion"]) > 0:
        axioms = gci_datasets["role_inclusion"][:]
        for row in axioms:
            r_id, s_id = int(row[0].item()), int(row[1].item())
            for y_id in range(nb_elements):
                rho_r = crisp_membership(mod.role_depth(all_pos, all_pos[y_id], r_id))
                rho_s = crisp_membership(mod.role_depth(all_pos, all_pos[y_id], s_id))
                assert float(role_inclusion(rho_r, rho_s).max().item()) == 0.0
            h_samp, t_samp = sample_role_region(mod, r_id, generator=gen, n_samples=n_region_samples)
            if bool(h_samp.valid.any().item()):
                zx = h_samp.points[h_samp.valid]
                zy = t_samp.points[t_samp.valid]
                rho_r = crisp_membership(mod.role_depth(zx, zy, r_id))
                rho_s = crisp_membership(mod.role_depth(zx, zy, s_id))
                assert float(role_inclusion(rho_r, rho_s).max().item()) == 0.0

    # 10. Role chain: r o s <= t (all element triples + 2000 region samples)
    if "role_chain" in gci_datasets and len(gci_datasets["role_chain"]) > 0:
        axioms = gci_datasets["role_chain"][:]
        for row in axioms:
            r_id, s_id, t_id = int(row[0].item()), int(row[1].item()), int(row[2].item())
            for y_id in range(nb_elements):
                rho_xy_r = crisp_membership(mod.role_depth(all_pos, all_pos[y_id], r_id))
                for z_id in range(nb_elements):
                    rho_yz_s = crisp_membership(
                        mod.role_depth(all_pos[y_id], all_pos[z_id], s_id)
                    ).expand_as(rho_xy_r)
                    rho_xz_t = crisp_membership(mod.role_depth(all_pos, all_pos[z_id], t_id))
                    assert float(role_chain(rho_xy_r, rho_yz_s, rho_xz_t).max().item()) == 0.0
            hx, hy = sample_role_region(mod, r_id, generator=gen, n_samples=n_region_samples)
            _, tz = sample_role_region(mod, s_id, generator=gen, n_samples=n_region_samples)
            valid_chain = hx.valid & tz.valid
            if bool(valid_chain.any().item()):
                zx = hx.points[valid_chain]
                zy = hy.points[valid_chain]
                zz = tz.points[valid_chain]
                rr = crisp_membership(mod.role_depth(zx, zy, r_id))
                rs = crisp_membership(mod.role_depth(zy, zz, s_id))
                rt = crisp_membership(mod.role_depth(zx, zz, t_id))
                assert float(role_chain(rr, rs, rt).max().item()) == 0.0

    # 11. ABox class_assertion and object_property_assertion
    if "class_assertion" in gci_datasets and len(gci_datasets["class_assertion"]) > 0:
        axioms = gci_datasets["class_assertion"][:]
        for row in axioms:
            # ClassAssertionDataset columns: [ind_idx, class_idx]
            ind_id, c_id = int(row[0].item()), int(row[1].item())
            elem_id = nb_classes + ind_id
            assert class_assertion(elem_m[elem_id : elem_id + 1, c_id]).item() == 0.0

    if (
        "object_property_assertion" in gci_datasets
        and len(gci_datasets["object_property_assertion"]) > 0
    ):
        axioms = gci_datasets["object_property_assertion"][:]
        for row in axioms:
            # ObjectPropertyAssertionDataset columns: [ind_a, rel_id, ind_b]
            ind_a, r_id, ind_b = int(row[0].item()), int(row[1].item()), int(row[2].item())
            el_a = nb_classes + ind_a
            el_b = nb_classes + ind_b
            rho_ab = crisp_membership(mod.role_depth(all_pos[el_a], all_pos[el_b], r_id))
            assert role_assertion(rho_ab).item() == 0.0


def test_exact_limit_init_t3_fixture(t3_dataset: PathDataset, t3_cm: CanonicalModel) -> None:
    """Exact-limit construction on T3 fixture yields crisp 0 on all elements and 2000 region samples."""
    el_ds = ELDataset(
        t3_dataset.ontology,
        extended=True,
        use_cache=False,
        load_role_axioms=True,
    )
    el_ds.load()
    _assert_exact_limit_zero_hinges(el_ds, t3_cm, n_region_samples=2000, seed=42)


def test_exact_limit_init_roles_fixture(roles_dataset: PathDataset, roles_cm: CanonicalModel) -> None:
    """Exact-limit construction on roles.owl (GCI1_bot, GCI2, GCI3, role_inclusion, role_chain, unsat E) yields crisp 0."""
    el_ds = ELDataset(
        roles_dataset.ontology,
        extended=True,
        use_cache=False,
        load_role_axioms=True,
    )
    el_ds.load()
    _assert_exact_limit_zero_hinges(el_ds, roles_cm, n_region_samples=2000, seed=77)


def test_exact_limit_init_comprehensive_role_and_abox_theory() -> None:
    """Exact-limit construction on a theory covering every normal form (including gci0_bot, gci3_bot, ABox) yields crisp 0."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/full_role_theory")

    top = adapter.create_class(TOP)
    bot = adapter.create_class(BOTTOM)
    A = adapter.create_class(NS + "A")
    B = adapter.create_class(NS + "B")
    C = adapter.create_class(NS + "C")
    D = adapter.create_class(NS + "D")
    U = adapter.create_class(NS + "U")
    W = adapter.create_class(NS + "W")
    r = adapter.create_object_property(NS + "r")
    s = adapter.create_object_property(NS + "s")
    t = adapter.create_object_property(NS + "t")
    ind_a = adapter.create_individual(IND + "a")
    ind_b = adapter.create_individual(IND + "b")

    # GCI0: A <= B
    mgr.addAxiom(ont, adapter.create_subclass_of(A, B))
    # GCI1: A cap B <= C
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(adapter.create_object_intersection_of(A, B), C),
    )
    # GCI0_bot: U <= bot
    mgr.addAxiom(ont, adapter.create_subclass_of(U, bot))
    # GCI1_bot: C cap U <= bot
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(adapter.create_object_intersection_of(C, U), bot),
    )
    # GCI2: A <= exists r.D, W <= exists r.U (propagates dead from U to W!)
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(A, adapter.create_object_some_values_from(r, D)),
    )
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(W, adapter.create_object_some_values_from(r, U)),
    )
    # GCI3: exists s.D <= C
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(adapter.create_object_some_values_from(s, D), C),
    )
    # GCI3_bot: exists t.U <= bot
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(adapter.create_object_some_values_from(t, U), bot),
    )
    # Role inclusion & chain: r <= s, r o s <= t
    from java.util import Arrays

    mgr.addAxiom(ont, df.getOWLSubObjectPropertyOfAxiom(r, s))
    mgr.addAxiom(ont, df.getOWLSubPropertyChainOfAxiom(Arrays.asList([r, s]), t))
    # ABox assertions: A(a), D(b), r(a, b)
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(A, ind_a))
    mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(D, ind_b))
    mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(r, ind_a, ind_b))

    cm = CanonicalModel.from_ontology(ont)
    el_ds = ELDataset(ont, extended=True, use_cache=False, load_role_axioms=True)
    el_ds.load()
    _assert_exact_limit_zero_hinges(el_ds, cm, n_region_samples=2000, seed=101)


# ---------------------------------------------------------------------------
# 5. Samplers unit tests (element samplers, region samplers, adversarial steps)
# ---------------------------------------------------------------------------


def test_element_and_region_samplers_and_adversarial_clipping() -> None:
    """Element and region samplers obey bounds, validity masks, and adversarial box clipping."""
    gen = th.Generator().manual_seed(2026)
    lhs = th.tensor([2, 4, 1], dtype=th.long)
    batch_el = th.tensor([0, 5], dtype=th.long)
    sampled = sample_elements(lhs, nb_elements=10, n_random=3, generator=gen, batch_elements=batch_el)
    assert sampled.shape == (3, 1 + 3 + 2)
    assert th.equal(sampled[:, 0], lhs)
    assert th.equal(sampled[:, -2:], batch_el.unsqueeze(0).expand(3, 2))
    assert int(sampled.min().item()) >= 0 and int(sampled.max().item()) < 10

    fillers = sample_filler_elements(th.tensor([1, 3]), nb_elements=8, n_random=2, generator=gen, batch_size=4)
    assert fillers.shape == (4, 4)
    assert fillers[:, :2].tolist() == [[1, 3]] * 4

    # Region sampler on active vs inactive components and adversarial ascent
    reg = Regions(
        cen=th.tensor([[[1.0, 2.0], [5.0, 5.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        w=th.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        active=th.tensor([[True, False], [False, False]]),
    )
    s0 = sample_active_component(reg, generator=gen, n_adv=0)
    assert s0.valid.tolist() == [True, False]
    assert bool((s0.points[0] > s0.lower[0]).all().item())
    assert bool((s0.points[0] < s0.upper[0]).all().item())

    # Adversarial ascent pushes point toward upper corner of [0.5, 1.5] x [1.5, 2.5] while staying clipped
    s_adv = sample_active_component(
        reg,
        generator=gen,
        n_adv=3,
        hinge_fn=lambda z: z.sum(dim=-1),
        step_size=1.0,
    )
    assert s_adv.points[0].tolist() == pytest.approx([1.5, 2.5])
    assert s_adv.points[1].tolist() == [0.0, 0.0]

    # Intersection sampler: row 0 overlaps on [0.5, 1.0]^2, row 1 disjoint
    reg_a = Regions(
        cen=th.tensor([[[0.5, 0.5]], [[0.0, 0.0]]]),
        w=th.tensor([[[0.5, 0.5]], [[0.2, 0.2]]]),
        active=th.tensor([[True], [True]]),
    )
    reg_b = Regions(
        cen=th.tensor([[[1.0, 1.0]], [[3.0, 3.0]]]),
        w=th.tensor([[[0.5, 0.5]], [[0.2, 0.2]]]),
        active=th.tensor([[True], [True]]),
    )
    s_inter = sample_component_intersection(reg_a, reg_b, generator=gen)
    assert s_inter.valid.tolist() == [True, False]
    assert s_inter.lower[0].tolist() == pytest.approx([0.5, 0.5])
    assert s_inter.upper[0].tolist() == pytest.approx([1.0, 1.0])
    assert bool((s_inter.points[0] >= 0.5).all() and (s_inter.points[0] <= 1.0).all())


def test_encoder_hook_and_set_tau() -> None:
    """GroundedELModule encoder hook computes individual positions and set_tau updates temperature."""
    feats = th.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    enc = nn.Linear(3, 2, bias=False)
    with th.no_grad():
        enc.weight.copy_(th.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    mod = GroundedELModule(
        nb_elements=5,
        nb_classes=3,
        nb_rels=1,
        embed_dim=2,
        element_features=feats,
        encoder=enc,
    )
    pos_ind = mod.positions([3, 4])
    assert th.allclose(pos_ind, th.tensor([[1.0, 2.0], [4.0, 5.0]]))

    mod.set_tau(0.05)
    assert float(mod.tau.item()) == pytest.approx(0.05)


def test_argument_validation() -> None:
    """Constructor and function argument validations reject invalid inputs."""
    with pytest.raises(ValueError, match="nb_elements"):
        GroundedELModule(nb_elements=2, nb_classes=4)
    with pytest.raises(ValueError, match="agg"):
        GroundedELModule(nb_elements=4, nb_classes=4, agg="invalid")
    with pytest.raises(ValueError, match="tau"):
        GroundedELModule(nb_elements=4, nb_classes=4, tau=0.0)
    with pytest.raises(ValueError, match="top_id"):
        GroundedELModule(nb_elements=4, nb_classes=4, top_id=1, bot_id=1)

    mod = GroundedELModule(nb_elements=4, nb_classes=4, n_components=2)
    with pytest.raises(ValueError, match="n_components >= nb_elements"):
        mod.exact_limit_init(th.zeros((4, 4), dtype=th.bool), edges=[])
    with pytest.raises(ValueError, match="eps"):
        minimality(th.tensor([0.5]), eps=-0.1)
    with pytest.raises(TypeError, match="Generator"):
        sample_elements(th.tensor([0]), 4, 1, generator=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6. AST test (R1): no reasoner or CanonicalModel imports/calls in grounded
# ---------------------------------------------------------------------------


def test_no_reasoner_or_canonical_model_in_grounded_ast() -> None:
    """No module under mowl/nn/el/grounded/ or mowl/models/grounded/ may import or call CanonicalModel, reasoners, or entails/joint_satisfiable (except evaluate_model in mowl/models/grounded/model.py)."""
    repo_root = Path(__file__).resolve().parents[2]
    nn_grounded_dir = repo_root / "mowl" / "nn" / "el" / "grounded"
    models_grounded_dir = repo_root / "mowl" / "models" / "grounded"
    nn_files = sorted(nn_grounded_dir.glob("*.py"))
    model_files = sorted(models_grounded_dir.glob("*.py"))
    assert len(nn_files) >= 4
    assert len(model_files) >= 2

    forbidden_import_substrings = (
        "mowl.geometric",
        "mowl.reasoning",
        "CanonicalModel",
        "Completion",
        "ELReasoner",
        "MOWLReasoner",
        "ElkReasoner",
    )
    forbidden_identifiers = {
        "CanonicalModel",
        "Completion",
        "CompletionResult",
        "ELReasoner",
        "MOWLReasoner",
        "ElkReasoner",
        "ElkReasonerFactory",
        "entails",
        "joint_satisfiable",
    }

    def _check_subtree(root: ast.AST, path: Path, allow_evaluate_model: bool) -> None:
        allowlisted_nodes: set[int] = set()
        if allow_evaluate_model:
            for n in ast.walk(root):
                if isinstance(n, ast.FunctionDef) and n.name == "evaluate_model":
                    for sub in ast.walk(n):
                        allowlisted_nodes.add(id(sub))

        for node in ast.walk(root):
            in_eval = id(node) in allowlisted_nodes
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in forbidden_import_substrings:
                        assert bad not in alias.name, (
                            f"Forbidden import '{alias.name}' in {path.name}:{node.lineno}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                if in_eval and mod_name == "mowl.geometric.completion":
                    for alias in node.names:
                        assert alias.name == "CanonicalModel", (
                            f"Only CanonicalModel may be imported in evaluate_model ({path.name}:{node.lineno})"
                        )
                    continue
                for bad in forbidden_import_substrings:
                    assert bad not in mod_name, (
                        f"Forbidden from-import '{mod_name}' in {path.name}:{node.lineno}"
                    )
                for alias in node.names:
                    assert alias.name not in forbidden_identifiers, (
                        f"Forbidden imported symbol '{alias.name}' in {path.name}:{node.lineno}"
                    )
            elif isinstance(node, ast.Name):
                if in_eval and node.id == "CanonicalModel":
                    continue
                assert node.id not in forbidden_identifiers, (
                    f"Forbidden identifier '{node.id}' in {path.name}:{node.lineno}"
                )
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_identifiers, (
                    f"Forbidden attribute/method '{node.attr}' in {path.name}:{node.lineno}"
                )

    for path in nn_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _check_subtree(tree, path, allow_evaluate_model=False)

    for path in model_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _check_subtree(
            tree,
            path,
            allow_evaluate_model=(path.name == "model.py"),
        )
        if path.name == "model.py":
            forbidden_reseed_keys = {
                "gci0",
                "gci1",
                "gci2",
                "gci3",
                "gci0_bot",
                "gci1_bot",
                "gci3_bot",
                "role_inclusion",
                "role_chain",
                "object_property_assertion",
            }
            for n in ast.walk(tree):
                if isinstance(n, ast.FunctionDef) and n.name == "reseed_idle":
                    for sub in ast.walk(n):
                        if (
                            isinstance(sub, ast.Constant)
                            and isinstance(sub.value, str)
                            and sub.value in forbidden_reseed_keys
                        ):
                            raise AssertionError(
                                f"reseed_idle must not access axiom table {sub.value!r} ({path.name}:{sub.lineno})"
                            )


def test_geometric_region_hinges_zero_on_containing_and_nonzero_grad_at_small_tau() -> None:
    """Geometric *_region hinges are 0 on containing boxes and > 0 with nonzero superclass gradient at tau=1e-3."""
    mod = GroundedELModule(
        nb_elements=5,
        nb_classes=5,
        nb_rels=3,
        embed_dim=2,
        n_components=1,
        n_role_components=1,
        tau=1e-3,
    )
    with th.no_grad():
        # Class 0 (A): [0.0, 1.0]^2 -> center (0.5, 0.5), w = 0.5
        # Class 1 (B_contain): [-0.5, 1.5]^2 -> center (0.5, 0.5), w = 1.0 (contains A)
        # Class 2 (B_viol): [2.0, 3.0]^2 -> center (2.5, 2.5), w = 0.5 (far from A, |depth| >> tau=1e-3)
        # Class 3 (A2_inter): [0.2, 1.2]^2 -> overlaps A on [0.2, 1.0]^2
        mod.class_centers[0, 0] = th.tensor([0.5, 0.5])
        mod.class_centers[1, 0] = th.tensor([0.5, 0.5])
        mod.class_centers[2, 0] = th.tensor([2.5, 2.5])
        mod.class_centers[3, 0] = th.tensor([0.7, 0.7])
        mod.class_raw_widths[0, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))
        mod.class_raw_widths[1, 0] = th.log(th.expm1(th.tensor([1.0, 1.0])))
        mod.class_raw_widths[2, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))
        mod.class_raw_widths[3, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))

        # Role 0 (r_contain): head [-0.5, 1.5]^2, tail [-0.5, 1.5]^2
        # Role 1 (r_viol): head [2.0, 3.0]^2, tail [2.0, 3.0]^2
        # Role 2 (s_mid): head [-0.5, 1.5]^2, tail [-0.5, 1.5]^2
        mod.role_head_centers[0, 0] = th.tensor([0.5, 0.5])
        mod.role_tail_centers[0, 0] = th.tensor([0.5, 0.5])
        mod.role_head_raw_widths[0, 0] = th.log(th.expm1(th.tensor([1.0, 1.0])))
        mod.role_tail_raw_widths[0, 0] = th.log(th.expm1(th.tensor([1.0, 1.0])))

        mod.role_head_centers[1, 0] = th.tensor([2.5, 2.5])
        mod.role_tail_centers[1, 0] = th.tensor([2.5, 2.5])
        mod.role_head_raw_widths[1, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))
        mod.role_tail_raw_widths[1, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))

        mod.role_head_centers[2, 0] = th.tensor([0.5, 0.5])
        mod.role_tail_centers[2, 0] = th.tensor([0.5, 0.5])
        mod.role_head_raw_widths[2, 0] = th.log(th.expm1(th.tensor([1.0, 1.0])))
        mod.role_tail_raw_widths[2, 0] = th.log(th.expm1(th.tensor([1.0, 1.0])))

    z_in_a = th.tensor([[0.5, 0.5]])
    y_in_a = th.tensor([[0.6, 0.6]])

    # 1. Containing configurations: all geometric region hinges are exact 0.0
    assert gci0_region(mod.depth(z_in_a, 1)).item() == 0.0
    assert gci1_region(mod.depth(z_in_a, 1)).item() == 0.0
    assert gci0_bot_region(mod.depth(z_in_a, 2)).item() == 0.0
    assert gci1_bot_region(mod.depth(z_in_a, 0), mod.depth(z_in_a, 2)).item() == 0.0
    assert gci2_region(mod.role_depth(z_in_a, y_in_a, 0)).item() == 0.0
    assert (
        gci3_region(
            mod.role_depth(z_in_a, y_in_a, 0),
            mod.depth(y_in_a, 0),
            mod.depth(z_in_a, 1),
        ).item()
        == 0.0
    )
    assert (
        gci3_bot_region(
            mod.role_depth(z_in_a, y_in_a, 0),
            mod.depth(y_in_a, 2),
        ).item()
        == 0.0
    )
    assert role_inclusion_region(mod.role_depth(z_in_a, y_in_a, 0)).item() == 0.0
    assert (
        role_chain_region(
            mod.role_depth(z_in_a, y_in_a, 0),
            mod.role_depth(y_in_a, z_in_a, 2),
            mod.role_depth(z_in_a, z_in_a, 0),
        ).item()
        == 0.0
    )

    # 2. Violating configurations at tau = 1e-3: positive loss AND nonzero gradient w.r.t. superclass/role box
    for loss_fn, target_param, target_idx in [
        (lambda: gci0_region(mod.depth(z_in_a, 2)), mod.class_centers, 2),
        (lambda: gci1_region(mod.depth(z_in_a, 2)), mod.class_centers, 2),
        (lambda: gci0_bot_region(mod.depth(z_in_a, 0)), mod.class_raw_widths, 0),
        (
            lambda: gci1_bot_region(mod.depth(z_in_a, 0), mod.depth(z_in_a, 3)),
            mod.class_centers,
            3,
        ),
        (lambda: gci2_region(mod.role_depth(z_in_a, y_in_a, 1)), mod.role_head_centers, 1),
        (
            lambda: gci3_region(
                mod.role_depth(z_in_a, y_in_a, 0),
                mod.depth(y_in_a, 0),
                mod.depth(z_in_a, 2),
            ),
            mod.class_centers,
            2,
        ),
        (
            lambda: gci3_bot_region(
                mod.role_depth(z_in_a, y_in_a, 0),
                mod.depth(y_in_a, 0),
            ),
            mod.class_raw_widths,
            0,
        ),
        (
            lambda: role_inclusion_region(mod.role_depth(z_in_a, y_in_a, 1)),
            mod.role_head_centers,
            1,
        ),
        (
            lambda: role_chain_region(
                mod.role_depth(z_in_a, y_in_a, 0),
                mod.role_depth(y_in_a, z_in_a, 2),
                mod.role_depth(z_in_a, z_in_a, 1),
            ),
            mod.role_head_centers,
            1,
        ),
    ]:
        mod.zero_grad()
        val = loss_fn()
        assert float(val.item()) > 0.0
        val.backward()
        assert target_param.grad is not None
        grad_norm = float(target_param.grad[target_idx].abs().sum().item())
        assert grad_norm > 0.0


def test_component_extremes_and_region_sample_cap() -> None:
    """component_extremes returns 2^d corners for d <= 4, 2d axis-extremes for d > 4, and region samplers cap at n_samples."""
    lo_3d = th.zeros((2, 3))
    hi_3d = th.ones((2, 3))
    ext_3d = component_extremes(lo_3d, hi_3d)
    assert ext_3d.shape == (2, 8, 3)
    assert bool(((ext_3d > 0.0) & (ext_3d < 1.0)).all().item())

    lo_6d = th.zeros((2, 6))
    hi_6d = th.ones((2, 6))
    ext_6d = component_extremes(lo_6d, hi_6d)
    assert ext_6d.shape == (2, 12, 6)
    assert bool(((ext_6d > 0.0) & (ext_6d < 1.0)).all().item())

    gen = th.Generator().manual_seed(11)
    mod = GroundedELModule(nb_elements=4, nb_classes=4, nb_rels=1, embed_dim=2, n_components=2)
    # 3 axiom rows with n_samples=4 -> capped at 4 samples per row = 12 total points
    s_cls = sample_class_region(mod, th.tensor([0, 1, 2]), generator=gen, n_samples=4)
    assert s_cls.points.shape == (12, 2)
    s_pair = sample_pair_region(
        mod, th.tensor([0, 1]), th.tensor([0, 1]), generator=gen, n_samples=4
    )
    assert s_pair.points.shape == (8, 2)


def test_sample_inside_elements_and_direct_dead_param() -> None:
    """sample_inside_elements selects elements crisp-inside LHS classes and GroundedELModule.dead() has unit gradient."""
    mod = GroundedELModule(
        nb_elements=5,
        nb_classes=4,
        nb_rels=0,
        embed_dim=2,
        n_components=1,
    )
    with th.no_grad():
        # Elements 0 and 4 inside class 0's box at [0, 0] with half-width 0.5; others outside
        mod.elem_pos.weight.copy_(
            th.tensor(
                [
                    [0.0, 0.0],
                    [3.0, 3.0],
                    [5.0, 5.0],
                    [7.0, 7.0],
                    [0.1, -0.1],
                ]
            )
        )
        mod.class_centers.fill_(10.0)
        mod.class_centers[0, 0] = th.tensor([0.0, 0.0])
        mod.class_raw_widths.fill_(float(th.log(th.expm1(th.tensor(0.5))).item()))

    gen = th.Generator().manual_seed(99)
    in_el = sample_inside_elements(mod, th.tensor([0, 1]), n_inside_elements=32, generator=gen)
    assert in_el.shape == (2, 2)
    assert sorted(in_el[0].tolist()) == [0, 4]

    comb = sample_elements(
        th.tensor([0, 1]),
        nb_elements=5,
        n_random=1,
        generator=gen,
        inside_elements=in_el,
    )
    assert comb.shape == (2, 1 + 1 + 2)
    assert th.equal(comb[:, -2:], in_el)

    # Direct dead parameterisation in [0, 1]: initial value 0.0 and unit gradient
    assert th.all(mod.dead() == 0.0)
    mod.zero_grad()
    mod.dead().sum().backward()
    assert mod.dead_logits.grad is not None
    assert th.allclose(mod.dead_logits.grad, th.ones_like(mod.dead_logits))


def test_reparametrised_region_sampling_lhs_gradients_and_sgd_step() -> None:
    """Reparametrised region samples give nonzero LHS centre/width gradients and one SGD step with RHS fixed reduces loss."""
    # 1. GCI1: A cap B <= C with C fixed (requires_grad=False) and A cap B sticking out of C
    mod = GroundedELModule(
        nb_elements=3,
        nb_classes=3,
        nb_rels=2,
        embed_dim=2,
        n_components=1,
        n_role_components=1,
        tau=0.05,
    )
    mod_fixed = GroundedELModule(
        nb_elements=3,
        nb_classes=3,
        nb_rels=2,
        embed_dim=2,
        n_components=1,
        n_role_components=1,
        tau=0.05,
    )
    mod_fixed.requires_grad_(False)

    with th.no_grad():
        # C (class 2 in mod_fixed): [-0.5, 0.5]^2 (center (0, 0), w (0.5, 0.5)), fixed
        mod_fixed.class_centers[2, 0] = th.tensor([0.0, 0.0])
        mod_fixed.class_raw_widths[2, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))
        # Also copy C into mod for adversarial hinge evaluation
        mod.class_centers[2, 0] = th.tensor([0.0, 0.0])
        mod.class_raw_widths[2, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))

        # A (class 0): [0.0, 1.2] x [0.2, 1.4] -> center (0.6, 0.8), w = (0.6, 0.6)
        # B (class 1): [0.2, 1.4] x [0.0, 1.2] -> center (0.8, 0.6), w = (0.6, 0.6)
        # Intersection A cap B = [0.2, 1.2]^2, sticking out of C = [-0.5, 0.5]^2 in the +x, +y quadrant
        mod.class_centers[0, 0] = th.tensor([0.6, 0.8])
        mod.class_raw_widths[0, 0] = th.log(th.expm1(th.tensor([0.6, 0.6])))
        mod.class_centers[1, 0] = th.tensor([0.8, 0.6])
        mod.class_raw_widths[1, 0] = th.log(th.expm1(th.tensor([0.6, 0.6])))

    def _eval_gci1_loss(seed_val: int) -> th.Tensor:
        g = th.Generator().manual_seed(seed_val)
        reg = sample_pair_region(
            mod,
            th.tensor([0]),
            th.tensor([1]),
            generator=g,
            n_samples=4,
            n_adv=2,
            hinge_fn=lambda z: gci1_region(mod_fixed.depth(z, 2)),
        )
        assert bool(reg.valid.all().item())
        return gci1_region(mod_fixed.depth(reg.points[reg.valid], 2)).mean()

    mod.zero_grad()
    loss_before_gci1 = _eval_gci1_loss(42)
    assert float(loss_before_gci1.item()) > 0.0
    loss_before_gci1.backward()

    assert mod.class_centers.grad is not None
    assert mod.class_raw_widths.grad is not None
    grad_ca = mod.class_centers.grad[0, 0].clone()
    grad_cb = mod.class_centers.grad[1, 0].clone()
    grad_wa = mod.class_raw_widths.grad[0, 0].clone()
    grad_wb = mod.class_raw_widths.grad[1, 0].clone()

    # Nonzero gradient w.r.t. both A's and B's centres and widths
    assert float(grad_ca.abs().sum().item()) > 0.0
    assert float(grad_cb.abs().sum().item()) > 0.0
    assert float(grad_wa.abs().sum().item()) > 0.0
    assert float(grad_wb.abs().sum().item()) > 0.0
    # C in mod_fixed has requires_grad=False (no grad)
    assert mod_fixed.class_centers.grad is None

    # Gradient sign on centres is positive, so -grad moves A cap B (at [0.2, 1.2]^2) towards C (at [-0.5, 0.5]^2)
    assert float((grad_ca + grad_cb).sum().item()) > 0.0

    with th.no_grad():
        inter_center_before = 0.5 * (
            th.maximum(
                mod.class_centers[0, 0] - nn.functional.softplus(mod.class_raw_widths[0, 0]),
                mod.class_centers[1, 0] - nn.functional.softplus(mod.class_raw_widths[1, 0]),
            )
            + th.minimum(
                mod.class_centers[0, 0] + nn.functional.softplus(mod.class_raw_widths[0, 0]),
                mod.class_centers[1, 0] + nn.functional.softplus(mod.class_raw_widths[1, 0]),
            )
        )
        lr = 0.1
        mod.class_centers[0, 0] -= lr * grad_ca
        mod.class_centers[1, 0] -= lr * grad_cb
        mod.class_raw_widths[0, 0] -= lr * grad_wa
        mod.class_raw_widths[1, 0] -= lr * grad_wb
        inter_center_after = 0.5 * (
            th.maximum(
                mod.class_centers[0, 0] - nn.functional.softplus(mod.class_raw_widths[0, 0]),
                mod.class_centers[1, 0] - nn.functional.softplus(mod.class_raw_widths[1, 0]),
            )
            + th.minimum(
                mod.class_centers[0, 0] + nn.functional.softplus(mod.class_raw_widths[0, 0]),
                mod.class_centers[1, 0] + nn.functional.softplus(mod.class_raw_widths[1, 0]),
            )
        )
    # Intersection centre moved closer to C's centre (0, 0)
    assert float(inter_center_after.norm().item()) < float(inter_center_before.norm().item())

    loss_after_gci1 = _eval_gci1_loss(42)
    assert float(loss_after_gci1.item()) < float(loss_before_gci1.item())

    # 2. GCI0: A <= B with B fixed (requires_grad=False) and A sticking out of B
    with th.no_grad():
        # B (class 1 in mod_fixed): [-0.5, 0.5]^2 (fixed)
        mod_fixed.class_centers[1, 0] = th.tensor([0.0, 0.0])
        mod_fixed.class_raw_widths[1, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))
        # A (class 0 in mod): [0.2, 1.2]^2 -> center (0.7, 0.7), w = (0.5, 0.5)
        mod.class_centers[0, 0] = th.tensor([0.7, 0.7])
        mod.class_raw_widths[0, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))

    def _eval_gci0_loss(seed_val: int) -> th.Tensor:
        g = th.Generator().manual_seed(seed_val)
        reg = sample_class_region(
            mod,
            th.tensor([0]),
            generator=g,
            n_samples=4,
            n_adv=2,
            hinge_fn=lambda z: gci0_region(mod_fixed.depth(z, 1)),
        )
        assert bool(reg.valid.all().item())
        return gci0_region(mod_fixed.depth(reg.points[reg.valid], 1)).mean()

    mod.zero_grad()
    loss_before_gci0 = _eval_gci0_loss(99)
    assert float(loss_before_gci0.item()) > 0.0
    loss_before_gci0.backward()

    grad_ca0 = mod.class_centers.grad[0, 0].clone()
    grad_wa0 = mod.class_raw_widths.grad[0, 0].clone()
    assert float(grad_ca0.abs().sum().item()) > 0.0
    assert float(grad_wa0.abs().sum().item()) > 0.0
    assert bool((grad_ca0 > 0.0).all().item())

    with th.no_grad():
        dist_before = float(mod.class_centers[0, 0].norm().item())
        mod.class_centers[0, 0] -= 0.1 * grad_ca0
        mod.class_raw_widths[0, 0] -= 0.1 * grad_wa0
        dist_after = float(mod.class_centers[0, 0].norm().item())
    assert dist_after < dist_before

    loss_after_gci0 = _eval_gci0_loss(99)
    assert float(loss_after_gci0.item()) < float(loss_before_gci0.item())

    # 3. Role inclusion r <= s with s fixed (requires_grad=False) and r sticking out of s
    with th.no_grad():
        mod_fixed.role_head_centers[1, 0] = th.tensor([0.0, 0.0])
        mod_fixed.role_tail_centers[1, 0] = th.tensor([0.0, 0.0])
        mod_fixed.role_head_raw_widths[1, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))
        mod_fixed.role_tail_raw_widths[1, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))

        mod.role_head_centers[0, 0] = th.tensor([0.7, 0.7])
        mod.role_tail_centers[0, 0] = th.tensor([0.7, 0.7])
        mod.role_head_raw_widths[0, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))
        mod.role_tail_raw_widths[0, 0] = th.log(th.expm1(th.tensor([0.5, 0.5])))

    mod.zero_grad()
    g_role = th.Generator().manual_seed(7)
    h_s, t_s = sample_role_region(
        mod,
        th.tensor([0]),
        generator=g_role,
        n_samples=4,
        n_adv=2,
        hinge_fn=lambda xy: role_inclusion_region(
            mod_fixed.role_depth(xy[:, :2], xy[:, 2:], 1)
        ),
    )
    loss_ri = role_inclusion_region(
        mod_fixed.role_depth(h_s.points[h_s.valid], t_s.points[t_s.valid], 1)
    ).mean()
    loss_ri.backward()
    assert mod.role_head_centers.grad is not None
    assert mod.role_tail_centers.grad is not None
    assert (
        float(mod.role_head_centers.grad[0, 0].abs().sum().item())
        + float(mod.role_tail_centers.grad[0, 0].abs().sum().item())
        > 0.0
    )


