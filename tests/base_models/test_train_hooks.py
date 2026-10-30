"""Unit tests for EmbeddingELModel training hooks and temperature annealing."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mowl.models import ELBE, MultiBoxEL, MultiELBE
from tests.datasetFactory import FamilyDataset


def test_train_hooks_called_in_order() -> None:
    ds = FamilyDataset()
    events: list[tuple[str, int, int]] = []

    class _HookedELBE(ELBE):
        def on_epoch_start(self, epoch: int, epochs: int) -> None:
            events.append(("start", epoch, epochs))

        def additional_training_loss(self, epoch: int) -> torch.Tensor | None:
            events.append(("extra", epoch, -1))
            if epoch % 2 == 0:
                return 0.01 * (self.module.class_embed.weight ** 2).mean()
            return None

    model = _HookedELBE(ds, embed_dim=8)
    model.train(
        epochs=3,
        validate_every=10,
        epoch_callback=lambda ep, m: events.append(("callback", ep, -1)),
    )
    expected = [
        ("start", 0, 3),
        ("extra", 0, -1),
        ("callback", 0, -1),
        ("start", 1, 3),
        ("extra", 1, -1),
        ("callback", 1, -1),
        ("start", 2, 3),
        ("extra", 2, -1),
        ("callback", 2, -1),
    ]
    assert events == expected


def test_train_hooks_absent_identical() -> None:
    ds = FamilyDataset()

    def _run(define_noop_hooks: bool) -> tuple[list[float], dict[str, torch.Tensor]]:
        np.random.seed(0)
        torch.manual_seed(0)
        if define_noop_hooks:
            class _NoopHooksELBE(ELBE):
                def on_epoch_start(self, epoch: int, epochs: int) -> None:
                    pass

                def additional_training_loss(self, epoch: int) -> None:
                    return None

            model = _NoopHooksELBE(ds, embed_dim=8, learning_rate=0.005)
        else:
            model = ELBE(ds, embed_dim=8, learning_rate=0.005)

        trajectory: list[float] = []

        def _record(ep: int, m: ELBE) -> None:
            with torch.no_grad():
                pos = m.module(m.training_datasets["gci0"][:], "gci0")
                trajectory.append(float(pos.mean().item()))

        np.random.seed(0)
        torch.manual_seed(0)
        model.train(epochs=3, validate_every=10, epoch_callback=_record)
        state = {k: v.detach().clone() for k, v in model.module.state_dict().items()}
        return trajectory, state

    traj_absent, state_absent = _run(False)
    traj_noop, state_noop = _run(True)
    assert len(traj_absent) == 3
    assert np.allclose(traj_absent, traj_noop, atol=1e-6, rtol=1e-5)
    for k in state_absent:
        assert torch.allclose(state_absent[k], state_noop[k], atol=1e-6, rtol=1e-5)


def test_tau_anneal_schedule() -> None:
    ds = FamilyDataset()
    tau_start = 1.2
    tau_end = 0.03
    epochs = 5

    for model_cls in (MultiBoxEL, MultiELBE):
        m_default = model_cls(ds, embed_dim=8, n_components=2)
        assert m_default.tau_end == pytest.approx(0.05, rel=1e-6)
        assert "tau" in m_default.module.state_dict()

        m = model_cls(
            ds,
            embed_dim=8,
            n_components=2,
            component_aggregation="soft",
            tau_start=tau_start,
            tau_end=tau_end,
            tau_anneal=True,
        )
        taus: list[float] = []
        m.train(
            epochs=epochs,
            validate_every=10,
            epoch_callback=lambda ep, mod: taus.append(float(mod.module.tau)),
        )
        assert len(taus) == epochs
        assert taus[0] == pytest.approx(tau_start, rel=1e-6)
        assert taus[-1] == pytest.approx(tau_end, rel=1e-6)
        for ep, val in enumerate(taus):
            expected = tau_start * ((tau_end / tau_start) ** (ep / (epochs - 1)))
            assert val == pytest.approx(expected, rel=1e-6)

        # Registered buffer tau is saved in state_dict and restored by load_state_dict
        saved_state = {k: v.clone() for k, v in m.module.state_dict().items()}
        assert float(saved_state["tau"].item()) == pytest.approx(tau_end, rel=1e-6)
        m_default.module.load_state_dict(saved_state)
        assert float(m_default.module.tau.item()) == pytest.approx(tau_end, rel=1e-6)

        # A subsequent train() call restarts the schedule from tau_start at epoch 0
        taus_restart: list[float] = []
        m.train(
            epochs=epochs,
            validate_every=10,
            epoch_callback=lambda ep, mod: taus_restart.append(float(mod.module.tau)),
        )
        assert taus_restart[0] == pytest.approx(tau_start, rel=1e-6)
        assert taus_restart[-1] == pytest.approx(tau_end, rel=1e-6)

        m_fixed = model_cls(
            ds,
            embed_dim=8,
            n_components=2,
            component_aggregation="soft",
            tau_start=tau_start,
            tau_end=tau_end,
            tau_anneal=False,
        )
        taus_fixed: list[float] = []
        m_fixed.train(
            epochs=3,
            validate_every=10,
            epoch_callback=lambda ep, mod: taus_fixed.append(float(mod.module.tau)),
        )
        assert all(t == pytest.approx(tau_start, rel=1e-6) for t in taus_fixed)

