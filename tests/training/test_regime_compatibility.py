"""A training regime is honoured or refused, never silently ignored."""

import pytest
import torch as th

from mowl.training import (ComposedRegime, EarlyStopping, HOOKS, LinearDecay,
                           PairwiseMarginRankingLoss, SCHEDULE_HOOKS, TrainingRegime,
                           unsupported_hooks)
from mowl.training.negatives import EitherSideCorruption
from tests.benchmarks.tiny import tiny_dataset


def test_a_regime_reports_the_hooks_it_overrides():
    assert TrainingRegime().overridden_hooks() == frozenset()
    assert "loss" in PairwiseMarginRankingLoss().overridden_hooks()
    assert "negatives" in EitherSideCorruption().overridden_hooks()
    schedule = LinearDecay().overridden_hooks()
    assert schedule and schedule <= SCHEDULE_HOOKS
    both = ComposedRegime(EitherSideCorruption(), LinearDecay()).overridden_hooks()
    assert both == EitherSideCorruption().overridden_hooks() | schedule
    assert set(both) <= set(HOOKS)


def test_the_generic_el_loop_takes_every_regime():
    from mowl.models import ELEmbeddings, OrderEmbeddings, DiskEmbeddings
    for cls in (ELEmbeddings, OrderEmbeddings, DiskEmbeddings):
        assert frozenset(cls.training_hooks) == frozenset(HOOKS)
        assert not unsupported_hooks(PairwiseMarginRankingLoss(), cls)


@pytest.mark.parametrize("name", ["FALCONModel", "E2R", "BoxE", "FaithEL", "VolumeMultiBoxEL",
                                  "DFALCModel", "CatE"])
def test_models_with_their_own_objective_refuse_loss_and_negative_regimes(name):
    import mowl.models as models
    cls = getattr(models, name)
    assert frozenset(cls.training_hooks) == frozenset()
    assert unsupported_hooks(PairwiseMarginRankingLoss(), cls) >= {"loss"}
    assert unsupported_hooks(LinearDecay(), cls)


def test_assigning_an_unsupported_regime_raises():
    from mowl.models import FaithEL, Query2Box
    with pytest.raises(TypeError, match="does not consult"):
        FaithEL(tiny_dataset()).training_regime = PairwiseMarginRankingLoss()
    model = Query2Box(tiny_dataset(), embed_dim=8)
    with pytest.raises(TypeError, match="would be ignored"):
        model.training_regime = EitherSideCorruption()
    with pytest.raises(TypeError, match="TrainingRegime"):
        model.training_regime = "linear"
    model.training_regime = None
    assert model.training_regime is None


def test_query_models_honour_schedule_regimes():
    from mowl.models import Query2Box
    assert frozenset(Query2Box.training_hooks) == SCHEDULE_HOOKS

    class Recorder(TrainingRegime):
        def __init__(self):
            super().__init__()
            self.rates, self.ended, self.losses = [], 0, []

        def optimizer(self):
            self.opt = th.optim.SGD(self.model.module.parameters(), lr=0.5)
            return self.opt

        def on_epoch_start(self, epoch, epochs):
            for group in self.opt.param_groups:
                group["lr"] = 0.5 / (epoch + 1)
            self.rates.append(self.opt.param_groups[0]["lr"])

        def on_epoch_end(self, epoch, losses):
            self.losses.append(losses["total"])

        def should_stop(self, epoch):
            return epoch == 2

        def on_train_end(self):
            self.ended += 1

    model = Query2Box(tiny_dataset(), embed_dim=8)
    regime = Recorder()
    model.training_regime = regime
    model.train(epochs=10, steps_per_epoch=2)
    assert regime.rates == [0.5, 0.25, 0.5 / 3]      # the regime's optimiser, stopped at epoch 2
    assert len(regime.losses) == 3 and regime.ended == 1
    assert model.loss_history == regime.losses

    model = Query2Box(tiny_dataset(), embed_dim=8)
    model.training_regime = LinearDecay()
    model.train(epochs=2, steps_per_epoch=1)
    assert len(model.loss_history) == 2


def test_early_stopping_is_exported():
    assert issubclass(EarlyStopping, TrainingRegime)
