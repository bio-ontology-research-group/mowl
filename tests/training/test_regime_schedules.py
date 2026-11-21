"""Learning-rate schedules, model selection and per-normal-form mini-batches."""
from unittest import TestCase

import torch as th

from mowl.training import ComposedRegime, TrainingRegime
from mowl.training.schedules import (StepDecay, ConstantThenCosine, CyclicLearningRate,
                                     ReduceOnPlateau)
from mowl.training.selection import ValidationSelection
from mowl.training.batching import PerNormalFormMiniBatches
from tests.training.stubs import StubModel


def _run_epochs(regime, epochs):
    for e in range(epochs):
        regime.on_epoch_start(e, epochs)


class TestSchedules(TestCase):

    def _attach(self, regime, lr=0.1):
        m = StubModel({"gci0": [[0, 1]]})
        m.learning_rate = lr
        regime.attach(m)
        regime.optimizer()
        return regime

    def test_step_decay(self):
        r = self._attach(StepDecay(milestones=[2, 4], gamma=0.1))
        _run_epochs(r, 6)
        expected = [0.1, 0.1, 0.01, 0.01, 0.001, 0.001]
        for a, b in zip(r.history, expected):
            self.assertAlmostEqual(a, b)

    def test_step_decay_matches_multisteplr(self):
        r = self._attach(StepDecay(milestones=[3]))
        p = th.nn.Parameter(th.zeros(1))
        opt = th.optim.SGD([p], lr=0.1)
        sched = th.optim.lr_scheduler.MultiStepLR(opt, milestones=[3], gamma=0.1)
        ref = []
        for _ in range(6):
            ref.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        _run_epochs(r, 6)
        for a, b in zip(r.history, ref):
            self.assertAlmostEqual(a, b)

    def test_cosine(self):
        r = self._attach(ConstantThenCosine(constant_fraction=0.5, min_lr=0.0), lr=1.0)
        _run_epochs(r, 6)
        self.assertEqual(r.history[:3], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(r.history[3], 1.0)
        self.assertAlmostEqual(r.history[4], 0.5)
        self.assertAlmostEqual(r.history[5], 0.0)

    def test_cyclic(self):
        r = self._attach(CyclicLearningRate(base_lr=0.1, max_lr=0.3, step_size_up=2))
        _run_epochs(r, 5)
        for a, b in zip(r.history, [0.1, 0.2, 0.3, 0.2, 0.1]):
            self.assertAlmostEqual(a, b, places=6)

    def test_plateau(self):
        values = iter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        r = self._attach(ReduceOnPlateau(metric=lambda m: next(values), patience=1,
                                         stop_patience=3))
        _run_epochs(r, 6)
        self.assertLess(r.history[-1], 0.1)
        self.assertEqual(r.stop_epoch, 3)

    def test_plateau_every_and_stop(self):
        values = iter([1.0] * 20)
        r = self._attach(ReduceOnPlateau(metric=lambda m: next(values), patience=1,
                                         stop_patience=1, every=3, stop=True))
        _run_epochs(r, 7)
        # monitored at epochs 3 and 6 only
        self.assertEqual(len(r.values), 2)
        self.assertEqual(r.stop_epoch, 5)
        self.assertTrue(r.should_stop(7))

    def test_early_stopping_stops_and_keeps_the_learning_rate(self):
        from mowl.training import EarlyStopping
        values = iter([1.0] * 20)
        r = self._attach(EarlyStopping(metric=lambda m: next(values), patience=3))
        _run_epochs(r, 6)
        self.assertEqual(r.key, "early-stopping")
        self.assertEqual(r.stop_epoch, 3)
        self.assertTrue(r.should_stop(6))
        self.assertAlmostEqual(r.history[-1], r.history[0])

    def test_plateau_every_from_batching(self):
        # every=None: monitor once per steps-per-epoch of the composed batching regime
        values = iter([1.0] * 20)
        plateau = ReduceOnPlateau(metric=lambda m: next(values), patience=1,
                                  stop_patience=1, every=None, stop=True)
        data = {"gci0": [[i, i + 1] for i in range(5)]}
        regime = ComposedRegime(PerNormalFormMiniBatches(batch_size=2), plateau)
        model = StubModel(data, batch_size=8)
        regime.attach(model)
        model.training_regime = regime
        model.training_datasets = regime.training_datasets(model.training_datasets)
        regime.optimizer()
        _run_epochs(regime, 7)
        self.assertEqual(len(plateau.values), 2)   # epochs 3 and 6
        self.assertEqual(plateau.stop_epoch, 5)
        self.assertTrue(regime.should_stop(7))

    def test_schedule_without_optimizer_raises(self):
        class Opt(TrainingRegime):
            def optimizer(self):
                return th.optim.SGD(self.model.module.parameters(), lr=1.0)
        regime = ComposedRegime(Opt(), StepDecay(1))
        regime.attach(StubModel({"gci0": [[0, 1]]}))
        regime.optimizer()
        with self.assertRaises(RuntimeError):
            regime.on_epoch_start(0, 1)


class TestSelection(TestCase):

    def test_keeps_best_state(self):
        m = StubModel({"gci0": [[0, 1]]})
        values = {1: 3.0, 2: 1.0, 3: 2.0}
        r = ValidationSelection(metric=lambda model: values[r._epochs_done - 1]
                                if r._epochs_done - 1 in values else 5.0)
        r.attach(m)
        for e in range(4):
            r.on_epoch_start(e, 4)
            with th.no_grad():
                m.module.scale.fill_(float(e + 1))   # "training"
        best = r.finalize()
        self.assertEqual(best, 2)
        self.assertAlmostEqual(float(m.module.scale), 2.0)

    def test_end_to_end_rank_selection(self):
        from mowl.models import ELEmbeddings
        from tests.datasetFactory import FamilyDataset
        from mowl.training.selection import ValidationRankSelection, validation_rank
        th.manual_seed(0)
        model = ELEmbeddings(FamilyDataset(), embed_dim=4, batch_size=64)
        # Family has no validation ontology: rank the training gci0 axioms instead.
        sel = ValidationRankSelection(every=1, gci_name="gci0", scoring="loss",
                                      split="train")
        model.training_regime = sel
        model.train(epochs=3, validate_every=100)
        best = sel.finalize()
        self.assertIn(best, (1, 2, 3))
        self.assertEqual(len(sel.history), 3)
        self.assertAlmostEqual(sel.best_value,
                               validation_rank(model, gci_name="gci0", scoring="loss",
                                               split="train"))


class TestBatching(TestCase):

    def test_random_batches(self):
        m = StubModel({"gci0": [[i, i + 1] for i in range(10)], "gci2": [[0, 0, 1]]},
                      batch_size=4)
        r = PerNormalFormMiniBatches(seed=0)
        r.attach(m)
        ds = r.training_datasets(m.training_datasets)
        self.assertEqual(len(ds["gci0"]), 10)          # full until the first epoch
        r.on_epoch_start(0, 1)
        self.assertEqual(len(ds["gci0"]), 4)
        self.assertEqual(len(ds["gci2"]), 1)            # small forms used in full
        batch = ds["gci0"][:]
        self.assertTrue(th.equal(batch, ds["gci0"].data))
        self.assertTrue(all(r_[1] == r_[0] + 1 for r_ in batch.tolist()))
        self.assertEqual(r.steps_per_epoch, 3)

    def test_cyclic_batches(self):
        m = StubModel({"gci0": [[i, i + 1] for i in range(10)]}, batch_size=4)
        r = PerNormalFormMiniBatches(order="cyclic")
        r.attach(m)
        ds = r.training_datasets(m.training_datasets)
        seen = []
        for e in range(4):
            r.on_epoch_start(e, 4)
            seen.append(ds["gci0"][:, 0].tolist())
        self.assertEqual(seen, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [0, 1, 2, 3]])

    def test_negatives_follow_the_batch(self):
        from mowl.training.negatives import FixedColumnCorruption
        m = StubModel({"gci0": [[i, i + 1] for i in range(10)]}, batch_size=3)
        regime = ComposedRegime(PerNormalFormMiniBatches(seed=0),
                                FixedColumnCorruption(columns={"gci0": 1}, seed=0))
        regime.attach(m)
        ds = regime.training_datasets(m.training_datasets)
        regime.on_epoch_start(0, 1)
        neg = regime.negatives("gci0", ds["gci0"])
        self.assertTrue(th.equal(neg[:, 0], ds["gci0"][:, 0]))

    def test_end_to_end(self):
        from mowl.models import ELEmbeddings
        from tests.datasetFactory import FamilyDataset
        th.manual_seed(0)
        model = ELEmbeddings(FamilyDataset(), embed_dim=4, batch_size=2)
        regime = PerNormalFormMiniBatches(seed=0)
        model.training_regime = regime
        model.train(epochs=3, validate_every=100)
        self.assertTrue(all(len(v) <= 2 for v in regime.views.values()))
