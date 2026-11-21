"""Tests for Task 16: corrupted-fact negatives and Lagrangian constraint weights in GroundedEL."""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch as th

from mowl.datasets import PathDataset
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.models import GroundedEL
from tests.geometric.helpers import t3_dataset


def test_backward_compatibility_default_arguments_and_validation(
    t3_dataset: PathDataset,
) -> None:
    """C.1: Default arguments preserve exact pre-Task-16 loss values, total formula, neg==0, and w_*==1."""
    # Pre-Task-16 CPU reference values on T3 (seed=42, generator seed=1234)
    ref_t3 = {
        "fact": 0.5172087550163269,
        "axioms": 0.42916926741600037,
        "region": 0.6834956407546997,
        "minimality": 0.0042516449466347694,
        "total": 1.6341252326965332,
    }
    model = GroundedEL(
        t3_dataset,
        embed_dim=4,
        n_components=2,
        n_role_components=2,
        epochs=0,
        seed=42,
    )
    assert model.neg_weight == 0.0
    assert model.neg_k == 1
    assert model.constraint_mode == "fixed"
    assert model.dual_rho == 1.0
    assert model.dual_max == 1e3
    assert model.dual_target == 0.0
    assert model.dual_weights == {"fact": 1.0, "axioms": 1.0, "region": 1.0}

    gen = th.Generator(device="cpu").manual_seed(1234)
    losses = model.compute_batch_losses(generator=gen)

    for k, expected in ref_t3.items():
        assert float(losses[k].item()) == pytest.approx(expected, abs=1e-7), (
            f"Mismatch on {k}: {losses[k].item()} vs {expected}"
        )

    expected_total = (
        losses["fact"]
        + losses["axioms"]
        + model.lambda_reg * losses["region"]
        + losses["minimality"]
    )
    assert th.equal(losses["total"], expected_total)
    assert losses["neg"].ndim == 0
    assert float(losses["neg"].item()) == 0.0

    for wk in ("w_fact", "w_axioms", "w_region"):
        assert wk in losses
        assert isinstance(losses[wk], th.Tensor)
        assert losses[wk].ndim == 0
        assert not losses[wk].requires_grad
        assert float(losses[wk].item()) == 1.0

    # Constructor validation for Task 16 arguments
    with pytest.raises(ValueError, match="neg_weight"):
        GroundedEL(t3_dataset, neg_weight=-0.1)
    with pytest.raises(ValueError, match="neg_k"):
        GroundedEL(t3_dataset, neg_k=0)
    with pytest.raises(ValueError, match="constraint_mode"):
        GroundedEL(t3_dataset, constraint_mode="invalid")
    with pytest.raises(ValueError, match="dual_rho"):
        GroundedEL(t3_dataset, dual_rho=-1.0)
    with pytest.raises(ValueError, match="dual_max"):
        GroundedEL(t3_dataset, dual_max=0.5)
    with pytest.raises(ValueError, match="dual_target"):
        GroundedEL(t3_dataset, dual_target=-0.01)


def test_corrupted_fact_negatives_loss_gradients_and_neg_k_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C.2: Toy KB (10 individuals, 2 roles, 20 role assertions): neg > 0, gradients reach translation boxes, neg_k scales corrupted rows."""
    class_index_dict = {BOTTOM: 0, TOP: 1, "C0": 2, "C1": 3}
    object_property_index_dict = {"r0": 0, "r1": 1}
    individual_index_dict = {f"i{k}": k for k in range(10)}

    # 20 role assertions across 10 individuals and 2 roles
    opa_rows = th.tensor(
        [[k % 10, (k // 10) % 2, (k + 1 + (k // 10)) % 10] for k in range(20)],
        dtype=th.long,
    )
    assert opa_rows.shape == (20, 3)

    model = GroundedEL.from_tables(
        class_index_dict=class_index_dict,
        object_property_index_dict=object_property_index_dict,
        individual_index_dict=individual_index_dict,
        object_property_assertion=opa_rows,
        role_mode="translation",
        neg_weight=1.0,
        neg_k=1,
        embed_dim=4,
        n_components=1,
        n_role_components=2,
        seed=13,
    )

    # Check helper _sample_corrupted_opa: only individual element IDs (>= nb_classes) are produced
    gen_check = th.Generator(device="cpu").manual_seed(99)
    x_neg, y_neg, r_neg = model._sample_corrupted_opa(opa_rows, gen_check)
    assert x_neg.shape == (20,)
    assert y_neg.shape == (20,)
    assert r_neg.shape == (20,)
    assert int(x_neg.min().item()) >= model.nb_classes
    assert int(x_neg.max().item()) < model.nb_classes + model.nb_inds
    assert int(y_neg.min().item()) >= model.nb_classes
    assert int(y_neg.max().item()) < model.nb_classes + model.nb_inds

    # Check neg > 0 and gradient reaches translation-box parameters
    model.module.zero_grad()
    losses = model.compute_batch_losses(generator=th.Generator(device="cpu").manual_seed(42))
    assert float(losses["neg"].item()) > 0.0
    expected_total = (
        losses["fact"]
        + losses["axioms"]
        + model.lambda_reg * losses["region"]
        + losses["minimality"]
        + model.neg_weight * losses["neg"]
    )
    assert th.allclose(losses["total"], expected_total, atol=1e-6)

    losses["neg"].backward()
    assert model.module.role_centers is not None
    assert model.module.role_raw_widths is not None
    assert model.module.role_centers.grad is not None
    assert model.module.role_raw_widths.grad is not None
    assert float(model.module.role_centers.grad.abs().sum().item()) > 0.0
    assert float(model.module.role_raw_widths.grad.abs().sum().item()) > 0.0

    # Check neg_k scales the number of corrupted rows both via helper and via monkeypatching role_membership
    for bsz, expected_base in ((8, 8), (4096, 20)):
        for k_mult in (1, 3, 5):
            m_k = GroundedEL.from_tables(
                class_index_dict=class_index_dict,
                object_property_index_dict=object_property_index_dict,
                individual_index_dict=individual_index_dict,
                object_property_assertion=opa_rows,
                role_mode="translation",
                neg_weight=1.0,
                neg_k=k_mult,
                batch_size=bsz,
                embed_dim=4,
                n_components=1,
                n_role_components=1,
                eps_role=0.0,
                seed=13,
            )
            g = th.Generator(device="cpu").manual_seed(77)
            xk, yk, rk = m_k._sample_corrupted_opa(opa_rows, g)
            assert xk.shape == (expected_base * k_mult,)
            assert yk.shape == (expected_base * k_mult,)
            assert rk.shape == (expected_base * k_mult,)

            call_sizes: List[int] = []
            orig_rm = m_k.module.role_membership

            def _tracked_rm(px: th.Tensor, py: th.Tensor, r: th.Tensor, *args, **kwargs):
                call_sizes.append(int(px.shape[0]))
                return orig_rm(px, py, r, *args, **kwargs)

            monkeypatch.setattr(m_k.module, "role_membership", _tracked_rm)
            m_k.compute_batch_losses(generator=th.Generator(device="cpu").manual_seed(77))
            # Call 0 is positive object_property_assertion (expected_base), Call 1 is corrupted negatives (expected_base * k_mult)
            assert call_sizes == [expected_base, expected_base * k_mult]


def test_dual_update_and_reset_dual_weights() -> None:
    """C.3: dual_update grows weights by dual_rho * excess, caps at dual_max, ignores <= dual_target, and is a no-op in fixed mode."""
    class_index_dict = {BOTTOM: 0, TOP: 1, "A": 2, "B": 3}
    gci0_t = th.tensor([[2, 3]], dtype=th.long)

    # 1. Fixed mode: weights stay 1.0 and dual_update returns them unchanged
    m_fixed = GroundedEL.from_tables(
        class_index_dict=class_index_dict,
        gci0=gci0_t,
        constraint_mode="fixed",
        dual_rho=2.0,
        dual_max=10.0,
        dual_target=0.05,
        embed_dim=4,
        seed=0,
    )
    w_fixed = m_fixed.dual_update({"fact": 0.5, "axioms": 1.2, "region": 3.0})
    assert w_fixed == {"fact": 1.0, "axioms": 1.0, "region": 1.0}
    assert m_fixed.dual_weights == {"fact": 1.0, "axioms": 1.0, "region": 1.0}

    # 2. Lagrangian mode: growth, target threshold, cap at dual_max, and reset
    m_lag = GroundedEL.from_tables(
        class_index_dict=class_index_dict,
        gci0=gci0_t,
        constraint_mode="lagrangian",
        dual_rho=2.5,
        dual_max=5.0,
        dual_target=0.1,
        embed_dim=4,
        seed=0,
    )
    assert m_lag.dual_weights == {"fact": 1.0, "axioms": 1.0, "region": 1.0}

    # Losses at or below dual_target=0.1 do not change weights
    w0 = m_lag.dual_update({"fact": 0.0, "axioms": 0.05, "region": 0.1})
    assert w0 == {"fact": 1.0, "axioms": 1.0, "region": 1.0}

    # Losses above dual_target=0.1 grow by dual_rho * (loss - dual_target)
    w1 = m_lag.dual_update({"fact": 0.3, "axioms": 0.5, "region": 1.1})
    assert w1["fact"] == pytest.approx(1.0 + 2.5 * (0.3 - 0.1))
    assert w1["axioms"] == pytest.approx(1.0 + 2.5 * (0.5 - 0.1))
    assert w1["region"] == pytest.approx(1.0 + 2.5 * (1.1 - 0.1))

    # Capped at dual_max=5.0
    w2 = m_lag.dual_update({"fact": 10.0, "axioms": 2.0, "region": 2.0})
    assert w2["fact"] == 5.0
    assert w2["axioms"] == 5.0
    assert w2["region"] == 5.0

    # Verify compute_batch_losses uses the updated dual weights in total and returns them
    losses_lag = m_lag.compute_batch_losses(generator=th.Generator(device="cpu").manual_seed(11))
    assert float(losses_lag["w_fact"].item()) == 5.0
    assert float(losses_lag["w_axioms"].item()) == 5.0
    assert float(losses_lag["w_region"].item()) == 5.0
    expected_total = (
        5.0 * losses_lag["fact"]
        + 5.0 * losses_lag["axioms"]
        + 5.0 * m_lag.lambda_reg * losses_lag["region"]
        + losses_lag["minimality"]
    )
    assert th.allclose(losses_lag["total"], expected_total, atol=1e-6)

    # Reset restores all weights to 1.0
    w_reset = m_lag.reset_dual_weights()
    assert w_reset == {"fact": 1.0, "axioms": 1.0, "region": 1.0}
    assert m_lag.dual_weights == {"fact": 1.0, "axioms": 1.0, "region": 1.0}


def test_lagrangian_vs_fixed_behavioural_role_inclusion_under_corrupted_negatives() -> None:
    """C.4: Toy KB with r <= s, r(a, b) and few individuals trained 300 steps with neg_weight=5, neg_k=4:
    fixed mode fails role_membership(p_a, p_b, s) > 0.5 while lagrangian mode satisfies it."""
    class_index_dict = {BOTTOM: 0, TOP: 1, "A": 2}
    object_property_index_dict = {"r": 0, "s": 1}
    # 2 individuals (a=0, b=1) so random head/tail corruptions of r(a, b) hit (a, r, b) with prob 0.5
    individual_index_dict = {"a": 0, "b": 1}
    role_inc = th.tensor([[0, 1]], dtype=th.long)  # r <= s
    opa = th.tensor([[0, 0, 1]], dtype=th.long)  # r(a, b)

    common_kwargs: Dict[str, object] = dict(
        class_index_dict=class_index_dict,
        object_property_index_dict=object_property_index_dict,
        individual_index_dict=individual_index_dict,
        role_inclusion=role_inc,
        object_property_assertion=opa,
        role_mode="translation",
        embed_dim=4,
        n_components=1,
        n_role_components=1,
        n_adv=0,
        epochs=300,
        learning_rate=0.05,
        tau0=0.2,
        tau1=0.05,
        neg_weight=5.0,
        neg_k=4,
        dual_rho=1.0,
        dual_max=1e3,
        dual_target=0.0,
        reseed_every=0,
        seed=0,
        log_every=0,
    )

    m_fixed = GroundedEL.from_tables(constraint_mode="fixed", **common_kwargs)  # type: ignore[arg-type]
    hist_fixed = m_fixed.train(epochs=300)
    assert len(hist_fixed) == 300
    assert hist_fixed[-1]["w_axioms"] == 1.0

    m_lag = GroundedEL.from_tables(constraint_mode="lagrangian", **common_kwargs)  # type: ignore[arg-type]
    hist_lag = m_lag.train(epochs=300)
    assert len(hist_lag) == 300
    assert hist_lag[-1]["w_axioms"] > 1.0

    el_a = m_fixed.nb_classes + 0
    el_b = m_fixed.nb_classes + 1
    s_id = 1

    with th.no_grad():
        pos_fixed = m_fixed.module.positions()
        rho_s_fixed = float(
            m_fixed.module.role_membership(
                pos_fixed[el_a], pos_fixed[el_b], s_id
            ).item()
        )

        pos_lag = m_lag.module.positions()
        rho_s_lag = float(
            m_lag.module.role_membership(
                pos_lag[el_a], pos_lag[el_b], s_id
            ).item()
        )

    # In fixed mode, corrupted negatives overwhelm unit-weight axiom loss so role_membership(p_a, p_b, s) > 0.5 fails
    assert not (rho_s_fixed > 0.5), (
        f"Expected fixed mode role_membership(p_a, p_b, s) <= 0.5, got {rho_s_fixed}"
    )

    # In lagrangian mode, dual ascent raises w_axioms/w_region until r(a, b) and r <= s hold
    assert rho_s_lag > 0.5, (
        f"Expected lagrangian mode role_membership(p_a, p_b, s) > 0.5, got {rho_s_lag}"
    )
