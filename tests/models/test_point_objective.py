"""Tests for Objective O3 (point and mixed modes) on MultiBoxEL and MultiELBE."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import Dataset, PathDataset
from mowl.geometric.normal_forms import TOP
from mowl.models import MultiBoxEL, MultiELBE
from mowl.nn import MultiBoxELModule, MultiELBEModule
from mowl.nn.el.pointloss import soft_box_membership, union_membership
from tests.geometric.helpers import fixture_path
from tests.geometric.test_abox import abox_kb  # noqa: F401
from tests.nn.test_multielbe_losses import _random_batch


def test_point_objective_tbox_only(tmp_path: Path, abox_kb) -> None:
    """On tk3 (no individuals), 3 epochs run under objective='point' and ind_offset grads are None."""
    ds = PathDataset(fixture_path("t3.owl"))
    ds.data_root = str(tmp_path / "cache_t3")

    for model_cls in (MultiBoxEL, MultiELBE):
        model = model_cls(
            ds,
            embed_dim=4,
            n_components=2,
            objective="point",
            points_per_epoch=50,
            points_seed=0,
            model_filepath=str(tmp_path / f"{model_cls.__name__}_pt.pt"),
        )
        model.train(epochs=3, validate_every=10)

        assert model.module.elem_embed is not None
        assert model.module.elem_embed.weight.grad is not None
        assert bool(
            (model.module.elem_embed.weight.grad.abs() > 0).any().item()
        )
        if model.module.ind_offset is not None:
            assert model.module.ind_offset.weight.grad is None

    # Also verify on an ABox dataset that ind_offset.weight.grad stays None while ind_embed gets grad
    abox_ds = Dataset(abox_kb)
    abox_ds.data_root = str(tmp_path / "cache_abox")
    for model_cls in (MultiBoxEL, MultiELBE):
        model = model_cls(
            abox_ds,
            embed_dim=4,
            n_components=2,
            objective="point",
            points_per_epoch=50,
            points_seed=0,
            model_filepath=str(tmp_path / f"{model_cls.__name__}_abox.pt"),
        )
        model.train(epochs=3, validate_every=10)
        assert model.module.ind_offset is not None
        assert model.module.ind_offset.weight.grad is None
        assert model.module.ind_embed is not None
        assert model.module.ind_embed.weight.grad is not None
        assert bool((model.module.ind_embed.weight.grad.abs() > 0).any().item())

        # Verify TOP is excluded from the volume penalty in additional_training_loss
        if TOP in model.named_class_index_dict:
            top_idx = model.named_class_index_dict[TOP]
            model.module.zero_grad()
            add_loss = model.additional_training_loss(0)
            assert add_loss is not None
            add_loss.backward()
            assert model.module.class_offset.weight.grad is not None
            assert torch.all(model.module.class_offset.weight.grad[top_idx] == 0.0)


def test_mixed_keeps_only_bottom_losses() -> None:
    """In 'mixed' mode only gci0_bot, gci1_bot, gci3_bot are non-zero; in 'point' mode all TBox losses are zero."""
    n_classes, n_rels, n_inds, d = 8, 3, 4, 4
    gen = torch.Generator().manual_seed(123)

    batches = {
        "gci0": _random_batch("gci0", 2, 16, n_classes, n_rels, n_inds, gen),
        "gci1": _random_batch("gci1", 3, 16, n_classes, n_rels, n_inds, gen),
        "gci2": _random_batch("gci2", 3, 16, n_classes, n_rels, n_inds, gen),
        "gci3": _random_batch("gci3", 3, 16, n_classes, n_rels, n_inds, gen),
        "gci0_bot": _random_batch("gci0_bot", 2, 16, n_classes, n_rels, n_inds, gen),
        "gci1_bot": _random_batch("gci1_bot", 3, 16, n_classes, n_rels, n_inds, gen),
        "gci3_bot": _random_batch("gci3_bot", 3, 16, n_classes, n_rels, n_inds, gen),
        "object_property_assertion": _random_batch(
            "object_property_assertion", 3, 16, n_classes, n_rels, n_inds, gen
        ),
    }

    for mod_cls in (MultiBoxELModule, MultiELBEModule):
        torch.manual_seed(7)
        mod_axiom = mod_cls(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            nb_inds=n_inds,
            embed_dim=d,
            n_components=2,
            objective="axiom",
        )
        torch.manual_seed(7)
        mod_mixed = mod_cls(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            nb_inds=n_inds,
            embed_dim=d,
            n_components=2,
            objective="mixed",
        )
        torch.manual_seed(7)
        mod_point = mod_cls(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            nb_inds=n_inds,
            embed_dim=d,
            n_components=2,
            objective="point",
        )

        non_bot_names = (
            "gci0",
            "gci1",
            "gci2",
            "gci3",
            "object_property_assertion",
        )
        bot_names = ("gci0_bot", "gci1_bot", "gci3_bot")

        # In training mode (module.training is True):
        mod_axiom.train()
        mod_mixed.train()
        mod_point.train()
        for gci in non_bot_names:
            out_ax = mod_axiom(batches[gci], gci)
            out_mx = mod_mixed(batches[gci], gci)
            out_pt = mod_point(batches[gci], gci)
            assert out_mx.shape == out_ax.shape
            assert out_pt.shape == out_ax.shape
            assert torch.all(out_mx == 0.0)
            assert torch.all(out_pt == 0.0)

        for gci in bot_names:
            out_ax = mod_axiom(batches[gci], gci)
            out_mx = mod_mixed(batches[gci], gci)
            out_pt = mod_point(batches[gci], gci)
            assert out_mx.shape == out_ax.shape
            assert out_pt.shape == out_ax.shape
            assert torch.allclose(out_mx, out_ax, atol=1e-6)
            assert bool((out_mx > 0.0).any().item())
            assert torch.all(out_pt == 0.0)

        # In eval mode (module.training is False): every GCI forward returns ordinary region scores
        mod_axiom.eval()
        mod_mixed.eval()
        mod_point.eval()
        for gci in non_bot_names + bot_names:
            out_ax_eval = mod_axiom(batches[gci], gci)
            out_mx_eval = mod_mixed(batches[gci], gci)
            out_pt_eval = mod_point(batches[gci], gci)
            assert torch.allclose(out_pt_eval, out_ax_eval, atol=1e-6)
            assert torch.allclose(out_mx_eval, out_ax_eval, atol=1e-6)


def test_point_mode_class_assertion_score_is_neg_log_m() -> None:
    """In 'point' and 'mixed' modes, class_assertion forward returns -log m."""
    n_classes, n_rels, n_inds, d = 6, 2, 4, 3
    data = torch.tensor([[0, 1], [1, 3], [2, 5], [3, 0]], dtype=torch.long)

    for mod_cls in (MultiBoxELModule, MultiELBEModule):
        for obj in ("point", "mixed"):
            mod = mod_cls(
                nb_ont_classes=n_classes,
                nb_rels=n_rels,
                nb_inds=n_inds,
                embed_dim=d,
                n_components=2,
                objective=obj,
                beta=0.1,
            )
            score_fwd = mod(data, "class_assertion")
            score_direct = mod.class_assertion_score_point(data)
            assert torch.allclose(score_fwd, score_direct, atol=1e-6)

            x = mod.ind_embed(data[:, 0])
            cen, off = mod.class_boxes(data[:, 1])
            m = union_membership(soft_box_membership(x, cen, off, beta=0.1))
            expected = -torch.log(m.clamp(min=1e-12))
            assert score_fwd.shape == (4,)
            assert torch.allclose(score_fwd, expected, atol=1e-6)
