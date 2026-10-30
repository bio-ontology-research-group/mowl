"""Unit tests for Prism losses and PrismModule."""
from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from mowl.nn import ELBEModule, PrismModule
import mowl.nn.el.prism.losses as PL


def _supports_to_center_inradius(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert support coordinates ``h = (h_1, h_2, h_3)`` to ``(center, inradius)``.

    With ``n_1 = (0, -1)``, ``n_2 = (sqrt(3)/2, 1/2)``, ``n_3 = (-sqrt(3)/2, 1/2)``
    and ``h_k = n_k . c + r``:
    - ``r = (h_1 + h_2 + h_3) / 3``
    - ``c_x = (h_2 - h_3) / sqrt(3)``
    - ``c_y = r - h_1``
    """
    h1 = h[..., 0]
    h2 = h[..., 1]
    h3 = h[..., 2]
    r = (h1 + h2 + h3) / 3.0
    cx = (h2 - h3) / math.sqrt(3.0)
    cy = r - h1
    center = torch.stack([cx, cy], dim=-1)
    return center, r.unsqueeze(-1)


def test_supports_sum_is_three_inradius() -> None:
    gen = torch.Generator().manual_seed(0)
    center = torch.randn(32, 2, generator=gen) * 5.0
    inradius = torch.abs(torch.randn(32, 1, generator=gen)) * 2.0
    h = PL.supports(center, inradius)
    assert h.shape == (32, 3)
    assert torch.allclose(h.sum(dim=-1), 3.0 * inradius.squeeze(-1), atol=1e-6)


def test_translation_support_values() -> None:
    gen = torch.Generator().manual_seed(1)
    center = torch.randn(16, 2, generator=gen)
    inradius = torch.abs(torch.randn(16, 1, generator=gen))
    t_vec = torch.tensor([0.75, -1.25])

    lhs = PL.supports(center + t_vec, inradius)
    rhs = PL.supports(center, inradius) + PL.NORMALS @ t_vec
    assert torch.allclose(lhs, rhs, atol=1e-6)

    # Also verify role translation in gci2_loss moves triangle via NORMALS @ t.
    mod = PrismModule(nb_ont_classes=2, nb_rels=1, embed_dim=4, margin=0.0)
    with torch.no_grad():
        mod.tri_center.weight[0] = torch.tensor([1.0, -0.5])
        mod.tri_inradius.weight[0] = torch.tensor([0.8])
        mod.box_center.weight[0] = torch.tensor([2.0, -1.0])
        mod.box_offset.weight[0] = torch.tensor([0.5, 0.5])

        mod.rel_embed.weight[0] = torch.tensor([0.75, -1.25, -0.5, 1.5])

        mod.tri_center.weight[1] = mod.tri_center.weight[0] + mod.rel_embed.weight[0, :2]
        mod.tri_inradius.weight[1] = mod.tri_inradius.weight[0]
        mod.box_center.weight[1] = mod.box_center.weight[0] + mod.rel_embed.weight[0, 2:]
        mod.box_offset.weight[1] = mod.box_offset.weight[0]

    loss = mod(torch.tensor([[0, 0, 1]]), "gci2")
    assert float(loss.item()) <= 1e-6


def test_inclusion_zero_iff_contained() -> None:
    # Contained: hc <= hd coordinatewise and box [bc - oc, bc + oc] inside [bd - od, bd + od]
    hc = torch.tensor([[1.0, 2.0, 0.5]])
    bc = torch.tensor([[0.0, 1.0]])
    oc = torch.tensor([[0.5, 0.5]])

    hd_good = torch.tensor([[1.5, 2.0, 1.0]])
    bd_good = torch.tensor([[0.2, 1.0]])
    od_good = torch.tensor([[0.8, 0.5]])

    score_good = PL.prism_inclusion_score(hc, bc, oc, hd_good, bd_good, od_good, margin=0.0)
    assert score_good.shape == (1,)
    assert float(score_good.item()) <= 1e-12

    # Triangle not contained (first support coordinate exceeds hd)
    hd_bad_tri = torch.tensor([[0.8, 2.5, 1.0]])
    score_bad_tri = PL.prism_inclusion_score(
        hc, bc, oc, hd_bad_tri, bd_good, od_good, margin=0.0
    )
    assert float(score_bad_tri.item()) > 0.1

    # Box not contained (first box coordinate exceeds [bd - od, bd + od])
    bd_bad_box = torch.tensor([[2.0, 1.0]])
    score_bad_box = PL.prism_inclusion_score(
        hc, bc, oc, hd_good, bd_bad_box, od_good, margin=0.0
    )
    assert float(score_bad_box.item()) > 0.1


def test_gci1_intersection_closed_form() -> None:
    h1 = torch.tensor([[3.0, 3.0, -1.0]])
    b1 = (torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0, 2.0]]))
    h2 = torch.tensor([[3.0, -1.0, 3.0]])
    b2 = (torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 1.0]]))

    h_inter, lower, upper = PL.prism_intersection(h1, b1, h2, b2)
    assert torch.allclose(h_inter, torch.minimum(h1, h2), atol=1e-6)
    assert torch.allclose(lower, torch.tensor([[-1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([[2.0, 2.0]]), atol=1e-6)

    # Nested in E -> 0 loss; not nested in E_bad -> positive loss
    mod = PrismModule(nb_ont_classes=4, nb_rels=1, embed_dim=4, margin=0.0)
    c1, r1 = _supports_to_center_inradius(h1[0])
    c2, r2 = _supports_to_center_inradius(h2[0])
    he_good = torch.tensor([3.5, 0.0, 0.0])  # >= min(h1, h2) = (3, -1, -1)
    ce_good, re_good = _supports_to_center_inradius(he_good)
    he_bad = torch.tensor([2.0, 2.0, 2.0])  # first coord 2.0 < 3.0
    ce_bad, re_bad = _supports_to_center_inradius(he_bad)

    with torch.no_grad():
        mod.tri_center.weight[0] = c1
        mod.tri_inradius.weight[0] = r1
        mod.box_center.weight[0] = b1[0][0]
        mod.box_offset.weight[0] = b1[1][0]

        mod.tri_center.weight[1] = c2
        mod.tri_inradius.weight[1] = r2
        mod.box_center.weight[1] = b2[0][0]
        mod.box_offset.weight[1] = b2[1][0]

        # E_good contains box intersection [-1, 2] x [0, 2] (center (0.5, 1.0), offset (1.5, 1.0))
        mod.tri_center.weight[2] = ce_good
        mod.tri_inradius.weight[2] = re_good
        mod.box_center.weight[2] = torch.tensor([0.5, 1.0])
        mod.box_offset.weight[2] = torch.tensor([2.0, 1.5])

        mod.tri_center.weight[3] = ce_bad
        mod.tri_inradius.weight[3] = re_bad
        mod.box_center.weight[3] = torch.tensor([0.5, 1.0])
        mod.box_offset.weight[3] = torch.tensor([2.0, 1.5])

    loss_good = mod(torch.tensor([[0, 1, 2]]), "gci1")
    loss_bad = mod(torch.tensor([[0, 1, 3]]), "gci1")
    assert float(loss_good.item()) <= 1e-6
    assert float(loss_bad.item()) > 0.5

    # Empty triangle intersection with nested box part gives > 0 at margin 0
    # h_disj1 = (1, 1, -1) (sum 1 >= 0) and h_disj2 = (-1, -1, 3) (sum 1 >= 0)
    # have min = (-1, -1, -1) (sum -3 < 0), which is <= he_good = (3.5, 0, 0).
    mod_empty_tri = PrismModule(nb_ont_classes=3, nb_rels=1, embed_dim=4, margin=0.0)
    cd1, rd1 = _supports_to_center_inradius(torch.tensor([1.0, 1.0, -1.0]))
    cd2, rd2 = _supports_to_center_inradius(torch.tensor([-1.0, -1.0, 3.0]))
    with torch.no_grad():
        mod_empty_tri.tri_center.weight[0] = cd1
        mod_empty_tri.tri_inradius.weight[0] = rd1
        mod_empty_tri.box_center.weight[0] = torch.tensor([0.0, 0.0])
        mod_empty_tri.box_offset.weight[0] = torch.tensor([0.5, 0.5])

        mod_empty_tri.tri_center.weight[1] = cd2
        mod_empty_tri.tri_inradius.weight[1] = rd2
        mod_empty_tri.box_center.weight[1] = torch.tensor([0.0, 0.0])
        mod_empty_tri.box_offset.weight[1] = torch.tensor([0.5, 0.5])

        mod_empty_tri.tri_center.weight[2] = ce_good
        mod_empty_tri.tri_inradius.weight[2] = re_good
        mod_empty_tri.box_center.weight[2] = torch.tensor([0.0, 0.0])
        mod_empty_tri.box_offset.weight[2] = torch.tensor([1.0, 1.0])

    loss_empty_tri = mod_empty_tri(torch.tensor([[0, 1, 2]]), "gci1")
    assert float(loss_empty_tri.item()) > 0.0
    assert torch.allclose(loss_empty_tri, torch.tensor([3.0]), atol=1e-6)


def test_gci1_bot_box_and_triangle_branches() -> None:
    """Test gci1_bot_loss across box-separation and triangle-separation branches."""
    mod = PrismModule(nb_ont_classes=6, nb_rels=1, embed_dim=4, margin=0.0)
    c_overlap, r_overlap = _supports_to_center_inradius(torch.tensor([1.0, 1.0, 1.0]))
    c_disj_a, r_disj_a = _supports_to_center_inradius(torch.tensor([1.0, 1.0, -1.0]))
    c_disj_b, r_disj_b = _supports_to_center_inradius(torch.tensor([-1.0, -1.0, 3.0]))

    with torch.no_grad():
        # Classes 0 and 1: overlapping triangles + boxes separated in coord 0 only, overlapping in coord 1
        mod.tri_center.weight[0] = c_overlap
        mod.tri_inradius.weight[0] = r_overlap
        mod.box_center.weight[0] = torch.tensor([0.0, 0.0])
        mod.box_offset.weight[0] = torch.tensor([0.5, 1.0])

        mod.tri_center.weight[1] = c_overlap
        mod.tri_inradius.weight[1] = r_overlap
        mod.box_center.weight[1] = torch.tensor([2.0, 0.0])
        mod.box_offset.weight[1] = torch.tensor([0.5, 1.0])

        # Classes 2 and 3: everything overlapping (triangles overlap and all box coords overlap)
        mod.tri_center.weight[2] = c_overlap
        mod.tri_inradius.weight[2] = r_overlap
        mod.box_center.weight[2] = torch.tensor([0.0, 0.0])
        mod.box_offset.weight[2] = torch.tensor([1.0, 1.0])

        mod.tri_center.weight[3] = c_overlap
        mod.tri_inradius.weight[3] = r_overlap
        mod.box_center.weight[3] = torch.tensor([0.5, 0.5])
        mod.box_offset.weight[3] = torch.tensor([1.0, 1.0])

        # Classes 4 and 5: triangles disjoint (sum min = -3 < 0) + boxes overlapping
        mod.tri_center.weight[4] = c_disj_a
        mod.tri_inradius.weight[4] = r_disj_a
        mod.box_center.weight[4] = torch.tensor([0.0, 0.0])
        mod.box_offset.weight[4] = torch.tensor([1.0, 1.0])

        mod.tri_center.weight[5] = c_disj_b
        mod.tri_inradius.weight[5] = r_disj_b
        mod.box_center.weight[5] = torch.tensor([0.0, 0.0])
        mod.box_offset.weight[5] = torch.tensor([1.0, 1.0])

    loss_one_box_sep = mod(torch.tensor([[0, 1]]), "gci1_bot")
    loss_all_overlap = mod(torch.tensor([[2, 3]]), "gci1_bot")
    loss_tri_disj = mod(torch.tensor([[4, 5]]), "gci1_bot")

    assert float(loss_one_box_sep.item()) == 0.0
    assert float(loss_all_overlap.item()) > 0.0
    assert float(loss_tri_disj.item()) == 0.0


def test_behavioural_gci3_bot_assertions_and_neg() -> None:
    """Behavioural zero-loss and positive-loss checks for gci3, gci3_bot, assertions, and neg paths."""
    mod = PrismModule(nb_ont_classes=4, nb_rels=2, nb_inds=3, embed_dim=4, margin=0.0)
    c_unit, r_unit = _supports_to_center_inradius(torch.tensor([1.0, 1.0, 1.0]))
    c_zero, r_zero = _supports_to_center_inradius(torch.tensor([0.0, 0.0, 0.0]))
    c_disj_a, r_disj_a = _supports_to_center_inradius(torch.tensor([1.0, 1.0, -1.0]))
    c_disj_b, r_disj_b = _supports_to_center_inradius(torch.tensor([-1.0, -1.0, 3.0]))

    with torch.no_grad():
        # Class 0: unit triangle at origin, box [-1, 1]^2
        mod.tri_center.weight[0] = c_unit
        mod.tri_inradius.weight[0] = r_unit
        mod.box_center.weight[0] = torch.tensor([0.0, 0.0])
        mod.box_offset.weight[0] = torch.tensor([1.0, 1.0])

        # Class 1: point triangle (inradius 0) at origin, box [0, 0]^2
        mod.tri_center.weight[1] = c_zero
        mod.tri_inradius.weight[1] = r_zero
        mod.box_center.weight[1] = torch.tensor([0.0, 0.0])
        mod.box_offset.weight[1] = torch.tensor([0.0, 0.0])

        # Class 2 & 3: disjoint triangles (sum min = -3), boxes far apart
        mod.tri_center.weight[2] = c_disj_a
        mod.tri_inradius.weight[2] = r_disj_a
        mod.box_center.weight[2] = torch.tensor([0.0, 0.0])
        mod.box_offset.weight[2] = torch.tensor([0.5, 0.5])

        mod.tri_center.weight[3] = c_disj_b
        mod.tri_inradius.weight[3] = r_disj_b
        mod.box_center.weight[3] = torch.tensor([5.0, 5.0])
        mod.box_offset.weight[3] = torch.tensor([0.5, 0.5])

        # Rel 0: zero translation; Rel 1: translation (1, -1, 2, -2)
        mod.rel_embed.weight[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        mod.rel_embed.weight[1] = torch.tensor([1.0, -1.0, 2.0, -2.0])

        # Ind 0: origin (inside Class 0); Ind 1: (1, -1, 2, -2) = Ind 0 + Rel 1; Ind 2: (10, 10, 10, 10) (outside Class 0)
        mod.ind_embed.weight[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        mod.ind_embed.weight[1] = torch.tensor([1.0, -1.0, 2.0, -2.0])
        mod.ind_embed.weight[2] = torch.tensor([10.0, 10.0, 10.0, 10.0])

    # gci3: exists r.C <= D (overlap of C - r with D)
    gci3_zero = mod(torch.tensor([[0, 0, 0]]), "gci3")
    gci3_pos = mod(torch.tensor([[0, 2, 3]]), "gci3")
    assert float(gci3_zero.item()) <= 1e-6
    assert float(gci3_pos.item()) > 0.0

    # gci3_bot: exists r.C <= bot (relu(3 * |r_C| + margin))
    gci3_bot_zero = mod(torch.tensor([[0, 1, 0]]), "gci3_bot")
    gci3_bot_pos = mod(torch.tensor([[0, 0, 0]]), "gci3_bot")
    assert float(gci3_bot_zero.item()) <= 1e-6
    assert float(gci3_bot_pos.item()) > 0.0

    # class_assertion: C(a)
    ca_zero = mod(torch.tensor([[0, 0]]), "class_assertion")
    ca_pos = mod(torch.tensor([[2, 0]]), "class_assertion")
    assert float(ca_zero.item()) <= 1e-6
    assert float(ca_pos.item()) > 0.0

    # object_property_assertion (neg=False): r(a, b)
    opa_zero = mod(torch.tensor([[0, 1, 1]]), "object_property_assertion", neg=False)
    opa_pos = mod(torch.tensor([[0, 1, 2]]), "object_property_assertion", neg=False)
    assert float(opa_zero.item()) <= 1e-6
    assert float(opa_pos.item()) > 0.0

    # gci2 (neg=True): 0 when C + r overlaps/equals D, > 0 when C + r is separated from D
    gci2_neg_zero = mod(torch.tensor([[0, 0, 0]]), "gci2", neg=True)
    gci2_neg_pos = mod(torch.tensor([[2, 0, 3]]), "gci2", neg=True)
    assert float(gci2_neg_zero.item()) <= 1e-6
    assert float(gci2_neg_pos.item()) > 0.0

    # object_property_assertion (neg=True): 0 when x_a + r == x_b, > 0 when separated
    opa_neg_zero = mod(torch.tensor([[0, 1, 1]]), "object_property_assertion", neg=True)
    opa_neg_pos = mod(torch.tensor([[0, 1, 2]]), "object_property_assertion", neg=True)
    assert float(opa_neg_zero.item()) <= 1e-6
    assert float(opa_neg_pos.item()) > 0.0


def _t3_axioms() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (gci0, gci1, gci1_bot) index tensors for T3 over classes 0..5.

    Classes: 0=X1, 1=X2, 2=X3, 3=A (X1 cap X2), 4=B (X2 cap X3), 5=C (X1 cap X3).
    """
    gci0 = torch.tensor(
        [[3, 0], [3, 1], [4, 1], [4, 2], [5, 0], [5, 2]], dtype=torch.long
    )
    gci1 = torch.tensor(
        [[0, 1, 3], [1, 2, 4], [0, 2, 5]], dtype=torch.long
    )
    gci1_bot = torch.tensor(
        [[3, 4], [4, 5], [3, 5], [3, 2], [4, 0], [5, 1]], dtype=torch.long
    )
    return gci0, gci1, gci1_bot


def _write_exact_t3_prism(mod: PrismModule) -> None:
    """Populate ``mod`` with the exact T3 support configuration and identical unit boxes."""
    h1 = torch.tensor([3.0, 3.0, -1.0])
    h2 = torch.tensor([3.0, -1.0, 3.0])
    h3 = torch.tensor([-1.0, 3.0, 3.0])
    ha = torch.minimum(h1, h2)  # (3, -1, -1), sum = 1
    hb = torch.minimum(h2, h3)  # (-1, -1, 3), sum = 1
    hc = torch.minimum(h1, h3)  # (-1, 3, -1), sum = 1
    all_h = torch.stack([h1, h2, h3, ha, hb, hc], dim=0)
    centers, inradii = _supports_to_center_inradius(all_h)
    with torch.no_grad():
        mod.tri_center.weight[:6].copy_(centers)
        mod.tri_inradius.weight[:6].copy_(inradii)
        mod.box_center.weight[:6].zero_()
        mod.box_offset.weight[:6].fill_(1.0)


def test_t3_exact_configuration_zero_loss_in_r2_plus_box() -> None:
    mod = PrismModule(nb_ont_classes=6, nb_rels=1, embed_dim=4, margin=0.0)
    _write_exact_t3_prism(mod)

    idx = torch.arange(6)
    h, bc, bo = mod.prism_params(idx)
    # Pairwise intersections A, B, C have support sum 1 >= 0
    assert torch.allclose(h[3:].sum(dim=-1), torch.ones(3), atol=1e-6)
    # Triple intersection has support sum -3 < 0
    triple_h = torch.minimum(torch.minimum(h[0], h[1]), h[2])
    assert torch.allclose(triple_h.sum(), torch.tensor(-3.0), atol=1e-6)

    gci0, gci1, gci1_bot = _t3_axioms()
    loss_0 = mod(gci0, "gci0")
    loss_1 = mod(gci1, "gci1")
    loss_1_bot = mod(gci1_bot, "gci1_bot")

    assert torch.all(loss_0 <= 1e-6)
    assert torch.all(loss_1 <= 1e-6)
    assert torch.all(loss_1_bot <= 1e-6)


@pytest.mark.slow
def test_prism_t3_locally_stable() -> None:
    """Stability check (not a learnability claim): PrismModule from the exact T₃
    configuration perturbed by 0.05 returns to loss < 1e-4 in 500 Adam steps.
    """
    gci0, gci1, gci1_bot = _t3_axioms()
    w_min = 0.05

    prism = PrismModule(nb_ont_classes=6, nb_rels=1, embed_dim=3, margin=0.0)
    _write_exact_t3_prism(prism)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in (
            prism.tri_center.weight,
            prism.tri_inradius.weight,
            prism.box_center.weight,
            prism.box_offset.weight,
        ):
            param.add_(0.05 * torch.randn(param.shape, generator=gen))

    opt_p = torch.optim.Adam(prism.parameters(), lr=0.005)
    sched_p = torch.optim.lr_scheduler.ExponentialLR(opt_p, gamma=0.985)
    for _ in range(500):
        opt_p.zero_grad()
        floor_p = (
            torch.relu(w_min - torch.abs(prism.tri_inradius.weight)).sum()
            + torch.relu(w_min - torch.abs(prism.box_offset.weight)).sum()
        )
        loss_p = (
            prism(gci0, "gci0").mean()
            + prism(gci1, "gci1").mean()
            + prism(gci1_bot, "gci1_bot").mean()
            + floor_p
        )
        loss_p.backward()
        opt_p.step()
        sched_p.step()

    with torch.no_grad():
        final_floor_p = (
            torch.relu(w_min - torch.abs(prism.tri_inradius.weight)).sum()
            + torch.relu(w_min - torch.abs(prism.box_offset.weight)).sum()
        )
        final_prism_loss = float(
            (
                prism(gci0, "gci0").mean()
                + prism(gci1, "gci1").mean()
                + prism(gci1_bot, "gci1_bot").mean()
                + final_floor_p
            ).item()
        )
    assert final_prism_loss < 1e-4, f"Expected prism T3 loss < 1e-4, got {final_prism_loss}"


@pytest.mark.slow
def test_elbe_t3_bound_best_of_restarts() -> None:
    """ELBEModule at d=2 with width floor w_min=0.05 cannot reach zero loss on T₃
    across 5 random restarts (seeds 0..4) of 3000 steps with Adam(lr=0.005) and
    ExponentialLR(0.985).

    The lower bound follows from Helly's theorem for axis-aligned boxes: since
    A ⊆ X1 ⊓ X2, B ⊆ X2 ⊓ X3, and C ⊆ X1 ⊓ X3 are non-empty, X1, X2, X3 are
    pairwise-intersecting axis-aligned boxes in ℝ^d and therefore share a common
    point x ∈ X1 ∩ X2 ∩ X3. With A = X1 ⊓ X2, that point satisfies x ∈ A ∩ X3,
    contradicting A ⊓ X3 ⊑ ⊥. Note that test_prism_t3_locally_stable is a local
    stability check from a perturbed exact configuration, not a random-init
    learnability claim.
    """
    gci0, gci1, gci1_bot = _t3_axioms()
    w_min = 0.05
    best_elbe_loss = float("inf")

    for seed in range(5):
        torch.manual_seed(seed)
        elbe = ELBEModule(nb_ont_classes=6, nb_rels=1, embed_dim=2, margin=0.0)
        opt_e = torch.optim.Adam(elbe.parameters(), lr=0.005)
        sched_e = torch.optim.lr_scheduler.ExponentialLR(opt_e, gamma=0.985)
        for _ in range(3000):
            opt_e.zero_grad()
            floor_e = torch.relu(w_min - torch.abs(elbe.class_offset.weight)).sum()
            loss_e = (
                elbe(gci0, "gci0").mean()
                + elbe(gci1, "gci1").mean()
                + elbe(gci1_bot, "gci1_bot").mean()
                + floor_e
            )
            loss_e.backward()
            opt_e.step()
            sched_e.step()

        with torch.no_grad():
            final_floor_e = torch.relu(w_min - torch.abs(elbe.class_offset.weight)).sum()
            final_loss = float(
                (
                    elbe(gci0, "gci0").mean()
                    + elbe(gci1, "gci1").mean()
                    + elbe(gci1_bot, "gci1_bot").mean()
                    + final_floor_e
                ).item()
            )
        if final_loss < best_elbe_loss:
            best_elbe_loss = final_loss

    assert best_elbe_loss > 1e-3, (
        f"Expected best-of-5 ELBE T3 loss > 1e-3 (Helly bound), got {best_elbe_loss:.6f}"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason="Optimisation gap: random-init PrismModule gets trapped in local minima on T3",
)
def test_prism_t3_from_random_init_gap() -> None:
    """Document the random-init optimisation gap for PrismModule on T₃ across
    seeds 0..4 with 3000 steps of Adam(lr=0.005) + ExponentialLR(0.985).
    """
    gci0, gci1, gci1_bot = _t3_axioms()
    w_min = 0.05
    best_prism_loss = float("inf")

    for seed in range(5):
        torch.manual_seed(seed)
        prism = PrismModule(nb_ont_classes=6, nb_rels=1, embed_dim=3, margin=0.0)
        opt_p = torch.optim.Adam(prism.parameters(), lr=0.005)
        sched_p = torch.optim.lr_scheduler.ExponentialLR(opt_p, gamma=0.985)
        for _ in range(3000):
            opt_p.zero_grad()
            floor_p = (
                torch.relu(w_min - torch.abs(prism.tri_inradius.weight)).sum()
                + torch.relu(w_min - torch.abs(prism.box_offset.weight)).sum()
            )
            loss_p = (
                prism(gci0, "gci0").mean()
                + prism(gci1, "gci1").mean()
                + prism(gci1_bot, "gci1_bot").mean()
                + floor_p
            )
            loss_p.backward()
            opt_p.step()
            sched_p.step()

        with torch.no_grad():
            final_floor_p = (
                torch.relu(w_min - torch.abs(prism.tri_inradius.weight)).sum()
                + torch.relu(w_min - torch.abs(prism.box_offset.weight)).sum()
            )
            final_loss = float(
                (
                    prism(gci0, "gci0").mean()
                    + prism(gci1, "gci1").mean()
                    + prism(gci1_bot, "gci1_bot").mean()
                    + final_floor_p
                ).item()
            )
        if final_loss < best_prism_loss:
            best_prism_loss = final_loss

    msg = (
        f"Best random-init PrismModule T3 final loss across 5 seeds (3000 steps): "
        f"{best_prism_loss:.6f} > 1e-4"
    )
    assert best_prism_loss <= 1e-4, msg


def test_empty_triangle_handling() -> None:
    # A triangle with sum(h) < 0 has zero triangle_empty_score and produces no NaN in inclusion
    h_empty = torch.tensor([[-1.0, -1.0, -1.0]], requires_grad=True)
    h_nonempty = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    bc = torch.tensor([[0.0, 0.0]], requires_grad=True)
    oc = torch.tensor([[1.0, 1.0]], requires_grad=True)
    bd = torch.tensor([[0.5, -0.5]], requires_grad=True)
    od = torch.tensor([[1.0, 1.0]], requires_grad=True)

    empty_score = PL.triangle_empty_score(h_empty)
    nonempty_score = PL.triangle_empty_score(h_nonempty)
    assert float(empty_score.item()) == 0.0
    assert float(nonempty_score.item()) == 3.0

    incl_sub_empty = PL.prism_inclusion_score(h_empty, bc, oc, h_nonempty, bd, od, margin=0.1)
    incl_super_empty = PL.prism_inclusion_score(h_nonempty, bc, oc, h_empty, bd, od, margin=0.1)
    assert torch.isfinite(incl_sub_empty).all()
    assert torch.isfinite(incl_super_empty).all()

    (incl_sub_empty + incl_super_empty).sum().backward()
    assert h_empty.grad is not None and torch.isfinite(h_empty.grad).all()
    assert h_nonempty.grad is not None and torch.isfinite(h_nonempty.grad).all()


def test_shapes_all_gcis() -> None:
    with pytest.raises(ValueError):
        PrismModule(nb_ont_classes=5, nb_rels=2, embed_dim=2)

    n_classes, n_rels, n_inds, d, B = 10, 3, 4, 6, 15
    mod = PrismModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        margin=0.1,
    )
    gen = torch.Generator().manual_seed(42)
    idx = torch.randint(0, n_classes, (B,), generator=gen)
    h, bc, bo = mod.prism_params(idx)
    assert h.shape == (B, 3)
    assert bc.shape == (B, d - 2)
    assert bo.shape == (B, d - 2)
    assert float(mod.regularization_loss().item()) == 0.0

    batches = {
        "gci0": torch.randint(0, n_classes, (B, 2), generator=gen),
        "gci0_bot": torch.randint(0, n_classes, (B, 2), generator=gen),
        "gci1": torch.randint(0, n_classes, (B, 3), generator=gen),
        "gci1_bot": torch.randint(0, n_classes, (B, 3), generator=gen),
        "gci2": torch.cat(
            [
                torch.randint(0, n_classes, (B, 1), generator=gen),
                torch.randint(0, n_rels, (B, 1), generator=gen),
                torch.randint(0, n_classes, (B, 1), generator=gen),
            ],
            dim=1,
        ),
        "gci3": torch.cat(
            [
                torch.randint(0, n_rels, (B, 1), generator=gen),
                torch.randint(0, n_classes, (B, 1), generator=gen),
                torch.randint(0, n_classes, (B, 1), generator=gen),
            ],
            dim=1,
        ),
        "gci3_bot": torch.cat(
            [
                torch.randint(0, n_rels, (B, 1), generator=gen),
                torch.randint(0, n_classes, (B, 1), generator=gen),
                torch.randint(0, n_classes, (B, 1), generator=gen),
            ],
            dim=1,
        ),
        "class_assertion": torch.cat(
            [
                torch.randint(0, n_inds, (B, 1), generator=gen),
                torch.randint(0, n_classes, (B, 1), generator=gen),
            ],
            dim=1,
        ),
        "object_property_assertion": torch.cat(
            [
                torch.randint(0, n_inds, (B, 1), generator=gen),
                torch.randint(0, n_rels, (B, 1), generator=gen),
                torch.randint(0, n_inds, (B, 1), generator=gen),
            ],
            dim=1,
        ),
    }

    for gci_name, batch in batches.items():
        out = mod(batch, gci_name, neg=False)
        assert out.shape == (B,), f"Expected ({B},) for {gci_name}, got {out.shape}"
        assert torch.isfinite(out).all()
        if gci_name in mod.neg_capable_gcis:
            out_neg = mod(batch, gci_name, neg=True)
            assert out_neg.shape == (B,)
            assert torch.isfinite(out_neg).all()

    mod_no_inds = PrismModule(nb_ont_classes=n_classes, nb_rels=n_rels, embed_dim=d)
    with pytest.raises(ValueError):
        mod_no_inds(batches["class_assertion"], "class_assertion")
    with pytest.raises(ValueError):
        mod_no_inds(batches["object_property_assertion"], "object_property_assertion")
