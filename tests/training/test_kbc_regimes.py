"""The training design of the faithfulness reference code (kbc) and its recipes."""
import os

import pytest
import torch as th

from mowl.training.kbc import DrawOnce, KBCCodeTraining, PooledAxiomMeanLoss
from mowl.training.negatives import FixedColumnCorruption
from tests.benchmarks.tiny import scratch_dir, tiny_dataset


def _model():
    from mowl.models import BoxSquaredEL
    model = BoxSquaredEL(tiny_dataset(), embed_dim=4, batch_size=32768,
                         model_filepath=os.path.join(scratch_dir(), "kbc.pt"))
    model.eval_gci_name = "gci0"
    return model


def test_pooled_axiom_mean():
    loss = PooledAxiomMeanLoss()
    loss.training_datasets({"gci0": th.zeros((3, 2)), "gci2": th.zeros((2, 3))})
    assert loss.num_positives == 5
    a = loss.combine("gci0", th.tensor([1.0, 2.0, 3.0]), None)
    b = loss.combine("gci2", th.tensor([1.0, 1.0]), th.tensor([4.0, 0.0]))
    # 5 positives + 2 negatives: (6 + 2 + 4) / 7 once the negatives have been seen
    assert float(b) == pytest.approx(6.0 / 7)
    assert float(a) == pytest.approx(6.0 / 5)
    assert loss.total == 7


def test_draw_once_reuses_the_first_draw():
    model = _model()
    regime = DrawOnce(FixedColumnCorruption(columns={"gci2": (2,)}, num_negatives=1,
                                            require_negative_loss=False, seed=3))
    regime.attach(model)
    data = model.training_datasets["gci2"]
    first = regime.negatives("gci2", data)
    assert th.equal(first, regime.negatives("gci2", data))
    assert regime.negatives("gci0", model.training_datasets["gci0"]) is NotImplemented


def test_kbc_code_training_trains_and_stops():
    model = _model()
    regime = KBCCodeTraining(patience=1, stop_patience=2)
    model.training_regime = regime
    assert [r.key for r in regime.regimes] == [
        "draw-negatives-once", "", "pooled-axiom-mean", "reduce-lr-on-plateau"]
    assert regime.negatives("gci0", model.training_datasets["gci0"]) is None
    negs = regime.negatives("gci2", model.training_datasets["gci2"])
    assert negs.shape == model.training_datasets["gci2"][:].shape
    assert regime.regimes[-1].early_stop
    gci0 = model.training_datasets["gci0"][:]
    before = float(model.module(gci0, "gci0").mean().detach())
    model.train(epochs=30)
    assert float(model.module(gci0, "gci0").mean().detach()) < before


def test_plateau_early_stop_ends_training():
    from mowl.training.schedules import ReduceOnPlateau
    model = _model()
    plateau = ReduceOnPlateau(metric=lambda m: 1.0, patience=1, stop_patience=3,
                              early_stop=True)
    model.training_regime = plateau
    model.train(epochs=100)
    assert plateau.stop_epoch is not None and len(plateau.history) < 10


def test_faithfulness_recipes():
    from mowl.benchmarks import get_recipe
    from mowl.catalog import get_regime
    r = get_recipe("faithfulness-pizza", "box2el")
    assert r.epochs == 6000 and r.training_regime == "kbc-code-training"
    assert r.model_kwargs["delta"] == 2 and r.model_kwargs["reg_factor"] == 0.05
    assert r.model_kwargs["batch_size"] == 32768
    assert len(get_recipe("faithfulness-goplus", "emelpp").model_kwargs) == 4
    for key in ("kbc-code-training", "pooled-axiom-mean", "draw-negatives-once"):
        assert get_regime(key).implemented
