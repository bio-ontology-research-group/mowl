"""Unit tests for mowl.nn.el.pointloss and exact T3 point scoring."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.datasets.el.points import PointSupervision
from mowl.geometric.completion import CanonicalModel
from mowl.models import MultiBoxEL, MultiELBE
from mowl.nn import MultiBoxELModule, MultiELBEModule
from mowl.nn.el.pointloss import (
    membership_bce,
    soft_box_membership,
    union_membership,
    volume_penalty,
)
from tests.geometric.helpers import _T3_EXACT_1D, _write_t3_exact, fixture_path


def test_union_membership_noisy_or() -> None:
    """soft_box_membership, union_membership (noisy-OR), membership_bce, and volume_penalty follow spec."""
    # 1. Noisy-OR exact formula: 1 - prod_i (1 - m_i)
    m = torch.tensor([[0.2, 0.5], [0.0, 0.0], [0.9, 0.9]], dtype=torch.float32)
    u = union_membership(m)
    expected = torch.tensor(
        [1.0 - 0.8 * 0.5, 0.0, 1.0 - 0.1 * 0.1], dtype=torch.float32
    )
    assert u.shape == (3,)
    assert torch.allclose(u, expected, atol=1e-6)

    # 2. soft_box_membership uses min over coordinates (no product decay across d)
    x = torch.tensor([[0.5, 0.5], [5.0, 5.0], [0.5, 0.2]], dtype=torch.float32)
    centers = torch.tensor(
        [
            [[0.5, 0.5], [2.5, 2.5]],
            [[0.5, 0.5], [2.5, 2.5]],
            [[0.5, 0.5], [2.5, 2.5]],
        ],
        dtype=torch.float32,
    )
    offsets = torch.full((3, 2, 2), 0.5, dtype=torch.float32)
    comp_m = soft_box_membership(x, centers, offsets, beta=0.1)
    assert comp_m.shape == (3, 2)
    sig5 = torch.sigmoid(torch.tensor(5.0)).item()
    sig2 = torch.sigmoid(torch.tensor(2.0)).item()
    assert comp_m[0, 0].item() == pytest.approx(sig5, rel=1e-5)
    assert comp_m[1, 0].item() < 1e-6
    assert comp_m[1, 1].item() < 1e-6
    # Asymmetric coordinate: min(sigmoid(5.0), sigmoid((0.5 - 0.3) / 0.1)) = sigmoid(2.0)
    assert comp_m[2, 0].item() == pytest.approx(sig2, rel=1e-5)

    # 3. membership_bce is positive and decreases as m approaches target
    bce_good = membership_bce(torch.tensor([0.95, 0.99]), target=1.0, weight=1.0)
    bce_bad = membership_bce(torch.tensor([0.10, 0.20]), target=1.0, weight=1.0)
    assert bce_good.item() > 0.0
    assert bce_good.item() < bce_bad.item()

    bce_weighted = membership_bce(torch.tensor([0.8]), target=0.0, weight=0.1)
    bce_unweighted = membership_bce(torch.tensor([0.8]), target=0.0, weight=1.0)
    assert bce_weighted.item() == pytest.approx(0.1 * bce_unweighted.item(), rel=1e-6)

    # 4. volume_penalty = mean over classes & components of mean_coord log(2|o| + 1e-6)
    offs = torch.tensor([[[0.5, 1.0], [0.25, 0.5]]], dtype=torch.float32)
    vp = volume_penalty(offs)
    expected_vp = torch.log(2.0 * offs + 1e-6).mean(dim=-1).mean()
    assert torch.allclose(vp, expected_vp, atol=1e-7)


def test_exact_t3_configuration_scores(tmp_path: Path) -> None:
    """Exact k=2 T3 boxes with elements at component centres give pos m >= 0.9 and hard neg m <= 0.1 at beta=0.1."""
    ds = PathDataset(fixture_path("t3.owl"))
    cm = CanonicalModel.from_dataset(ds)

    for model_cls in (MultiBoxEL, MultiELBE):
        model = model_cls(
            ds,
            embed_dim=1,
            n_components=2,
            margin=0.0,
            beta=0.1,
            model_filepath=str(tmp_path / f"{model_cls.__name__}_t3.pt"),
        )
        _write_t3_exact(model)

        ps = PointSupervision.from_canonical_model(
            cm,
            model.named_class_index_dict,
            model.individual_index_dict,
            model.object_property_index_dict,
        )
        model.module.set_point_supervision(ps)

        # Place each class element at component 0's centre of its own class
        with torch.no_grad():
            for elem_idx, elem_name in enumerate(ps.element_names):
                lo, hi = _T3_EXACT_1D[elem_name.split("#")[-1]][0]
                model.module.elem_embed.weight[elem_idx] = torch.tensor(
                    [(lo + hi) / 2.0]
                )

            pos_elem = ps.pos_pairs[:, 0]
            pos_cls = ps.pos_pairs[:, 1]
            x_pos = model.module.element_embeddings(pos_elem)
            c_pos, o_pos = model.module.class_boxes(pos_cls)
            m_pos = union_membership(
                soft_box_membership(x_pos, c_pos, o_pos, beta=0.1)
            )
            assert bool((m_pos >= 0.9).all().item()), (
                f"Expected all positive pairs m >= 0.9, min was {m_pos.min().item()}"
            )

            neg_elem = ps.hard_neg_pairs[:, 0]
            neg_cls = ps.hard_neg_pairs[:, 1]
            x_neg = model.module.element_embeddings(neg_elem)
            c_neg, o_neg = model.module.class_boxes(neg_cls)
            m_neg = union_membership(
                soft_box_membership(x_neg, c_neg, o_neg, beta=0.1)
            )
            assert bool((m_neg <= 0.1).all().item()), (
                f"Expected all hard neg pairs m <= 0.1, max was {m_neg.max().item()}"
            )

            stats = model.module.point_modelness(ps)
            assert stats["pos_frac"] == pytest.approx(1.0)
            assert stats["hardneg_frac"] == pytest.approx(1.0)


def test_point_role_loss_numeric_multielbe_and_multiboxel() -> None:
    """Hand-computed numeric tests for point_role_loss on MultiELBE and MultiBoxEL (class and individual targets)."""
    # 3 elements: 0 -> class 0, 1 -> class 1, 2 -> individual 0
    ps = PointSupervision(
        n_elements=3,
        element_names=["http://ex/C0", "http://ex/C1", "http://ex/ind0"],
        element_class_index=torch.tensor([0, 1, 0], dtype=torch.long),
        element_is_individual=torch.tensor([False, False, True], dtype=torch.bool),
        pos_pairs=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        hard_neg_pairs=torch.empty((0, 2), dtype=torch.long),
        role_pairs=torch.tensor([[0, 0, 1], [0, 0, 2]], dtype=torch.long),
        rng_seed=0,
    )

    beta = 0.1
    # --- 1. MultiELBE numeric check ---
    elbe = MultiELBEModule(
        nb_ont_classes=2,
        nb_rels=1,
        nb_inds=1,
        embed_dim=2,
        n_components=2,
        objective="point",
        beta=beta,
    )
    elbe.set_point_supervision(ps)
    assert elbe.ind_halfwidth == pytest.approx(5.0 * beta)

    with torch.no_grad():
        # Element 0 (class 0): x_0 = [1.0, 2.0]
        # Element 1 (class 1): x_1 = [0.0, 0.0]
        elbe.elem_embed.weight[0] = torch.tensor([1.0, 2.0])
        elbe.elem_embed.weight[1] = torch.tensor([0.0, 0.0])
        # Individual 0 (element 2): y_ind = [1.4, 1.8]
        elbe.ind_embed.weight[0] = torch.tensor([1.4, 1.8])
        # Role 0 translation: r_0 = [0.5, -0.3] -> x_0 + r_0 = [1.5, 1.7]
        elbe.rel_embed.weight[0] = torch.tensor([0.5, -0.3])
        # Class 1 boxes (k=2, d=2):
        # comp 0: center [1.4, 1.9], offset [0.4, 0.5] -> |o| - |x+r - c| = [0.4-0.1, 0.5-0.2] = [0.3, 0.3] -> min = sig(3.0)
        # comp 1: center [0.0, 0.0], offset [0.2, 0.2] -> |o| - |x+r - c| = [0.2-1.5, 0.2-1.7] = [-1.3, -1.5] -> min = sig(-15.0)
        elbe.class_embed.weight[1] = torch.tensor([1.4, 1.9, 0.0, 0.0])
        elbe.class_offset.weight[1] = torch.tensor([0.4, 0.5, 0.2, 0.2])

    # Hand computation for MultiELBE:
    # Triple 0 (class target C1): m_comp0 = sig(3.0), m_comp1 = sig(-15.0)
    m_c1_comp0 = torch.sigmoid(torch.tensor(3.0))
    m_c1_comp1 = torch.sigmoid(torch.tensor(-15.0))
    m_elbe_cls = 1.0 - (1.0 - m_c1_comp0) * (1.0 - m_c1_comp1)
    # Triple 1 (individual target ind0): virtual box center [1.4, 1.8], halfwidth 0.5
    # |x_0 + r_0 - y_ind| = |[1.5, 1.7] - [1.4, 1.8]| = [0.1, 0.1]
    # (0.5 - 0.1) / 0.1 = 4.0 -> m_elbe_ind = sig(4.0)
    m_elbe_ind = torch.sigmoid(torch.tensor(4.0))

    loss_elbe_cls = elbe.point_role_loss({"roles": torch.tensor([[0, 0, 1]], dtype=torch.long)})
    loss_elbe_ind = elbe.point_role_loss({"roles": torch.tensor([[0, 0, 2]], dtype=torch.long)})
    loss_elbe_both = elbe.point_role_loss({"roles": ps.role_pairs})

    assert torch.allclose(loss_elbe_cls, -torch.log(m_elbe_cls), atol=1e-6)
    assert torch.allclose(loss_elbe_ind, -torch.log(m_elbe_ind), atol=1e-6)
    expected_elbe_both = 0.5 * (-torch.log(m_elbe_cls) - torch.log(m_elbe_ind))
    assert torch.allclose(loss_elbe_both, expected_elbe_both, atol=1e-6)

    # Exact hit on individual target on MultiELBE: x_0 + r_0 == y_ind -> m = sig(5.0) >= 0.99
    with torch.no_grad():
        elbe.ind_embed.weight[0] = torch.tensor([1.5, 1.7])
    loss_elbe_exact_ind = elbe.point_role_loss({"roles": torch.tensor([[0, 0, 2]], dtype=torch.long)})
    m_exact_elbe = torch.exp(-loss_elbe_exact_ind).item()
    assert m_exact_elbe >= 0.99
    assert m_exact_elbe == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item(), rel=1e-5)

    # --- 2. MultiBoxEL numeric check ---
    boxel = MultiBoxELModule(
        nb_ont_classes=2,
        nb_rels=1,
        nb_inds=1,
        embed_dim=2,
        n_components=2,
        objective="point",
        beta=beta,
    )
    boxel.set_point_supervision(ps)
    assert boxel.ind_halfwidth == pytest.approx(5.0 * beta)

    with torch.no_grad():
        # Element points: x_0 = [1.0, 2.0], x_1 = [0.5, -0.5], ind_0 = [2.0, 1.0]
        boxel.elem_embed.weight[0] = torch.tensor([1.0, 2.0])
        boxel.elem_embed.weight[1] = torch.tensor([0.5, -0.5])
        boxel.ind_center.weight[0] = torch.tensor([2.0, 1.0])
        # Bumps: bump_classes[0] = [0.1, 0.2], bump_classes[1] = [-0.2, 0.3], bump_individuals[0] = [0.4, -0.1]
        boxel.bump_classes.weight[0] = torch.tensor([0.1, 0.2])
        boxel.bump_classes.weight[1] = torch.tensor([-0.2, 0.3])
        boxel.bump_individuals.weight[0] = torch.tensor([0.4, -0.1])
        # Role 0 head box: center [0.9, 2.1], offset [0.5, 0.5]
        # Role 0 tail box: center [0.8, -0.1], offset [0.6, 0.4]
        boxel.head_center.weight[0] = torch.tensor([0.9, 2.1])
        boxel.head_offset.weight[0] = torch.tensor([0.5, 0.5])
        boxel.tail_center.weight[0] = torch.tensor([0.8, -0.1])
        boxel.tail_offset.weight[0] = torch.tensor([0.6, 0.4])

    # Hand computation for MultiBoxEL Triple 0 (elem_x=0 class 0, elem_y=1 class 1):
    # x + b_y = [1.0, 2.0] + [-0.2, 0.3] = [0.8, 2.3]
    # |o_H| - |x + b_y - c_H| = [0.5 - 0.1, 0.5 - 0.2] = [0.4, 0.3] -> min = 0.3 -> m_H = sig(3.0)
    # y + b_x = [0.5, -0.5] + [0.1, 0.2] = [0.6, -0.3]
    # |o_T| - |y + b_x - c_T| = [0.6 - 0.2, 0.4 - 0.2] = [0.4, 0.2] -> min = 0.2 -> m_T = sig(2.0)
    m_boxel_cls = torch.sigmoid(torch.tensor(3.0)) * torch.sigmoid(torch.tensor(2.0))
    loss_boxel_cls = boxel.point_role_loss({"roles": torch.tensor([[0, 0, 1]], dtype=torch.long)})
    assert torch.allclose(loss_boxel_cls, -torch.log(m_boxel_cls), atol=1e-6)

    # Hand computation for MultiBoxEL Triple 1 (elem_x=0 class 0, elem_y=2 individual 0):
    # x + b_ind0 = [1.0, 2.0] + [0.4, -0.1] = [1.4, 1.9]
    # |o_H| - |x + b_ind0 - c_H| = [0.5 - 0.5, 0.5 - 0.2] = [0.0, 0.3] -> min = 0.0 -> m_H = sig(0.0) = 0.5
    # Individual target uses virtual box (y_ind0, ind_halfwidth=0.5) on translated tail centre c_T - b_x:
    # c_T - b_x = [0.8, -0.1] - [0.1, 0.2] = [0.7, -0.3]
    # ind_halfwidth - |(c_T - b_x) - y_ind0| = [0.5 - 1.3, 0.5 - 1.3] = [-0.8, -0.8] -> min = -0.8 -> m_T = sig(-8.0)
    m_boxel_ind = torch.sigmoid(torch.tensor(0.0)) * torch.sigmoid(torch.tensor(-8.0))
    loss_boxel_ind = boxel.point_role_loss({"roles": torch.tensor([[0, 0, 2]], dtype=torch.long)})
    assert torch.allclose(loss_boxel_ind, -torch.log(m_boxel_ind), atol=1e-6)

    # Zero out tail_offset so only ind_halfwidth = 5*beta can give m_T = sig(5.0) >= 0.99
    # when c_T - b_x == y_ind0 and x + b_ind0 == c_H:
    with torch.no_grad():
        boxel.head_center.weight[0] = torch.tensor([1.4, 1.9])
        boxel.head_offset.weight[0] = torch.tensor([0.6, 0.6])
        boxel.tail_center.weight[0] = torch.tensor([2.1, 1.2])
        boxel.tail_offset.weight[0] = torch.tensor([0.0, 0.0])

    m_target_elbe = elbe.target_membership(
        torch.tensor([[1.5, 1.7]]), torch.tensor([2], dtype=torch.long)
    )
    m_target_boxel = boxel.target_membership(
        torch.tensor([[2.0, 1.0]]), torch.tensor([2], dtype=torch.long)
    )
    assert m_target_elbe.item() >= 0.99
    assert m_target_boxel.item() >= 0.99
    assert m_target_boxel.item() == pytest.approx(
        torch.sigmoid(torch.tensor(5.0)).item(), rel=1e-5
    )

    loss_boxel_exact_ind = boxel.point_role_loss(
        {"roles": torch.tensor([[0, 0, 2]], dtype=torch.long)}
    )
    m_exact_boxel = torch.exp(-loss_boxel_exact_ind).item()
    assert m_exact_boxel >= 0.99
    assert m_exact_boxel == pytest.approx(
        (torch.sigmoid(torch.tensor(6.0)) * torch.sigmoid(torch.tensor(5.0))).item(),
        rel=1e-5,
    )
