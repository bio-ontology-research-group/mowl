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
from mowl.nn.el.pointloss import (
    membership_bce,
    soft_box_membership,
    union_membership,
    volume_penalty,
)
from tests.geometric.helpers import NS, fixture_path
from tests.models.test_multiboxel import _T3_EXACT_1D, _write_t3_exact


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

    # 2. soft_box_membership at box centre vs outside box
    x = torch.tensor([[0.5, 0.5], [5.0, 5.0]], dtype=torch.float32)
    centers = torch.tensor(
        [[[0.5, 0.5], [2.5, 2.5]], [[0.5, 0.5], [2.5, 2.5]]], dtype=torch.float32
    )
    offsets = torch.full((2, 2, 2), 0.5, dtype=torch.float32)
    comp_m = soft_box_membership(x, centers, offsets, beta=0.1)
    assert comp_m.shape == (2, 2)
    sig5_sq = torch.sigmoid(torch.tensor(5.0)).item() ** 2
    assert comp_m[0, 0].item() == pytest.approx(sig5_sq, rel=1e-5)
    assert comp_m[1, 0].item() < 1e-6
    assert comp_m[1, 1].item() < 1e-6

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
