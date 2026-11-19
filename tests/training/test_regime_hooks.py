"""The generic EL training loop honours every TrainingRegime hook, and is unchanged
without a regime."""
from unittest import TestCase

import torch as th

from mowl.models import ELEmbeddings
from mowl.training import TrainingRegime, ComposedRegime
from tests.datasetFactory import FamilyDataset


class RecordingRegime(TrainingRegime):
    """Overrides every hook and records the calls."""

    key = "recording"

    def __init__(self):
        super().__init__()
        self.calls = []
        self.attached_to = None

    def attach(self, model):
        super().attach(model)
        self.attached_to = model

    def training_datasets(self, datasets):
        self.calls.append("training_datasets")
        # train on gci0 only
        return {"gci0": datasets["gci0"]}

    def negatives(self, gci_name, gci_dataset):
        self.calls.append(("negatives", gci_name))
        return None

    def loss(self, gci_name, pos_scores, neg_scores):
        self.calls.append(("loss", gci_name, neg_scores is None))
        return th.mean(pos_scores) * 0.0 + 7.0

    def on_epoch_start(self, epoch, epochs):
        self.calls.append(("epoch", epoch, epochs))

    def additional_loss(self, epoch):
        self.calls.append(("additional", epoch))
        return th.tensor(1.0)

    def optimizer(self):
        self.calls.append("optimizer")
        self.opt = th.optim.SGD(self.model.module.parameters(), lr=0.0)
        return self.opt


class FallbackRegime(TrainingRegime):
    key = "fallback"


def _params(model):
    return [p.detach().clone() for p in model.module.parameters()]


class TestRegimeHooks(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _model(self):
        th.manual_seed(0)
        return ELEmbeddings(self.dataset, embed_dim=8, batch_size=64)

    def test_default_is_none(self):
        self.assertIsNone(self._model().training_regime)

    def test_setter_attaches_and_rejects_other_types(self):
        model = self._model()
        regime = RecordingRegime()
        model.training_regime = regime
        self.assertIs(regime.attached_to, model)
        self.assertIs(regime.model, model)
        with self.assertRaises(TypeError):
            model.training_regime = object()
        model.training_regime = None
        self.assertIsNone(model.training_regime)

    def test_every_hook_is_honoured(self):
        model = self._model()
        regime = RecordingRegime()
        model.training_regime = regime
        before = _params(model)
        model.train(epochs=2)
        calls = regime.calls
        self.assertEqual(calls.count("training_datasets"), 1)
        self.assertEqual(calls.count("optimizer"), 1)
        self.assertIn(("epoch", 0, 2), calls)
        self.assertIn(("epoch", 1, 2), calls)
        self.assertIn(("additional", 0), calls)
        # Only gci0 is trained on, negatives None -> loss sees no negatives
        self.assertEqual([c for c in calls if c[0] == "negatives"],
                         [("negatives", "gci0")] * 2)
        self.assertEqual([c for c in calls if c[0] == "loss"],
                         [("loss", "gci0", True)] * 2)
        # SGD with lr=0 from the regime was used: parameters did not move
        for a, b in zip(before, _params(model)):
            self.assertTrue(th.equal(a, b))

    def test_neutral_regime_equals_no_regime(self):
        """A regime returning the neutral default everywhere trains exactly like no regime."""
        import numpy as np
        results = []
        for regime in (None, FallbackRegime(), ComposedRegime(FallbackRegime())):
            model = self._model()
            if regime is not None:
                model.training_regime = regime
            np.random.seed(1)
            th.manual_seed(1)
            model.train(epochs=3)
            results.append(_params(model))
        for other in results[1:]:
            for a, b in zip(results[0], other):
                self.assertTrue(th.equal(a, b))

    def test_composed_precedence(self):
        first, second = RecordingRegime(), RecordingRegime()

        class Neg(TrainingRegime):
            def negatives(self, gci_name, gci_dataset):
                return "neg-from-second"

        composed = ComposedRegime(FallbackRegime(), Neg(), first)
        self.assertEqual(composed.negatives("gci0", None), "neg-from-second")
        composed2 = ComposedRegime(first, second)
        model = self._model()
        model.training_regime = composed2
        self.assertIs(first.model, model)
        self.assertIs(second.model, model)
        self.assertEqual(composed2.additional_loss(0).item(), 2.0)


class _MarkFilter(TrainingRegime):
    """Filter that records what it sees and replaces the first column by ``value``."""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.seen = []

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        self.seen.append((gci_name, None if negatives is None else negatives.clone()))
        if negatives is None:
            return None
        out = negatives.clone()
        out[:, 0] = self.value
        return out


class _ConstNegatives(TrainingRegime):
    def negatives(self, gci_name, gci_dataset):
        if gci_name != "gci2":
            return None
        data = gci_dataset[:]
        return data.clone()


class TestFilterNegativesHook(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _model(self):
        th.manual_seed(0)
        return ELEmbeddings(self.dataset, embed_dim=8, batch_size=64)

    def test_filter_applied_to_model_default_negatives(self):
        model = self._model()
        filt = _MarkFilter(0)
        model.training_regime = filt
        model.train(epochs=1)
        seen = dict(filt.seen)
        # ELEm samples negatives for gci2 by default; the filter received them
        self.assertIsNotNone(seen["gci2"])
        self.assertEqual(len(seen["gci2"]), len(model.training_datasets["gci2"]))
        # and None for forms without negatives
        self.assertIsNone(seen["gci0"])

    def test_filter_applied_to_regime_negatives_and_chained_in_order(self):
        model = self._model()
        first, second = _MarkFilter(1), _MarkFilter(2)
        model.training_regime = ComposedRegime(_ConstNegatives(), first, second)
        model.train(epochs=1)
        pos = model.training_datasets["gci2"][:]
        got_first = dict(first.seen)["gci2"]
        got_second = dict(second.seen)["gci2"]
        # first filter sees the regime's negatives, second sees the first's output
        self.assertTrue(th.equal(got_first, pos))
        self.assertTrue((got_second[:, 0] == 1).all())


class _StopAt(TrainingRegime):
    def __init__(self, stop_epoch):
        super().__init__()
        self.stop_epoch = stop_epoch
        self.epochs_seen = []
        self.ended = 0

    def on_epoch_start(self, epoch, epochs):
        self.epochs_seen.append(epoch)

    def should_stop(self, epoch):
        return epoch == self.stop_epoch

    def on_train_end(self):
        self.ended += 1


class _ScoreCapture(TrainingRegime):
    def __init__(self, like_positives):
        super().__init__()
        self.score_negatives_like_positives = like_positives
        self.captured = {}

    def negatives(self, gci_name, gci_dataset):
        return gci_dataset[:].clone() if gci_name == "gci0" else None

    def loss(self, gci_name, pos_scores, neg_scores):
        if gci_name == "gci0":
            self.captured["pos"] = pos_scores.detach().clone()
            self.captured["neg"] = neg_scores.detach().clone()
        return th.mean(pos_scores)


class TestStopEndAndNegativeScoring(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _model(self):
        th.manual_seed(0)
        return ELEmbeddings(self.dataset, embed_dim=8, batch_size=64)

    def test_should_stop_and_on_train_end(self):
        model = self._model()
        regime = _StopAt(1)
        model.training_regime = regime
        model.train(epochs=5)
        self.assertEqual(regime.epochs_seen, [0, 1])
        self.assertEqual(regime.ended, 1)

    def test_on_train_end_without_early_stop(self):
        model = self._model()
        regime = _StopAt(-1)
        model.training_regime = regime
        model.train(epochs=3)
        self.assertEqual(regime.epochs_seen, [0, 1, 2])
        self.assertEqual(regime.ended, 1)

    def test_composed_should_stop_asks_every_regime(self):
        a, b = _StopAt(-1), _StopAt(0)
        model = self._model()
        model.training_regime = ComposedRegime(a, b)
        model.train(epochs=4)
        self.assertEqual(a.epochs_seen, [0])
        self.assertEqual((a.ended, b.ended), (1, 1))

    def test_score_negatives_like_positives(self):
        for flag in (False, True):
            model = self._model()
            # ELEm with DELE losses: neg=True gives a different score than the positive form
            model.module.enable_all_negative_losses()
            regime = _ScoreCapture(flag)
            model.training_regime = regime
            model.train(epochs=1)
            same = th.equal(regime.captured["pos"], regime.captured["neg"])
            self.assertEqual(same, flag)
