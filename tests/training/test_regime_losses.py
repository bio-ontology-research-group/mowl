"""Loss-combination regimes: hand-computed values on tiny tensors, re-scoring of negatives,
equivalence with built-in model losses, composition and an end-to-end run."""
import math
from unittest import TestCase

import torch as th
import torch.nn.functional as F

from mowl.training import ComposedRegime
from mowl.training.losses import (SumLoss, MSETargetLoss, PairwiseMarginRankingLoss,
                                  LogSigmoidLoss, BPRLoss, LogSumExpContrastiveLoss,
                                  PearsonCorrelationLoss, ProbabilisticBCELoss,
                                  NormalFormFrequencyWeighting)
from mowl.training.negatives import FixedColumnCorruption
from tests.training.stubs import StubModel

POS = th.tensor([[0.5], [1.0]])
NEG = th.tensor([[2.0], [3.0], [0.0], [4.0]])  # K=2 blocks of n=2: pos0 -> 2, 0; pos1 -> 3, 4


def _module_mode(regime):
    """Attach to a stub and use the scores passed in (no re-scoring)."""
    regime.neg_scores = "module"
    regime.attach(StubModel({"gci0": [[0, 1]]}))
    return regime


class TestValues(TestCase):

    def test_sum(self):
        r = SumLoss(neg_weight=0.5)
        r.attach(StubModel({"gci0": [[0, 1]]}))
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)), 0.75 + 0.5 * 2.25)

    def test_sum_equals_model_compute_loss(self):
        m = StubModel({"gci0": [[0, 1]]})
        r = SumLoss()
        r.attach(m)
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)),
                               float(m.compute_loss(POS, NEG)))

    def test_mse(self):
        r = MSETargetLoss(neg_target=2.0)
        r.attach(StubModel({"gci0": [[0, 1]]}))
        expected = (0.25 + 1.0) / 2 + (0 + 1 + 4 + 4) / 4
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)), expected, places=6)

    def test_mse_mean_first_equals_mowl_elbe(self):
        """MSETargetLoss(neg_target=1, reduce='mean_first') is mOWL ELBE's compute_loss."""
        from mowl.models import ELBE
        from tests.datasetFactory import FamilyDataset
        elbe = ELBE(FamilyDataset(), embed_dim=4)
        r = MSETargetLoss(neg_target=1.0, reduce="mean_first")
        r.attach(elbe)
        pos, neg = th.rand(7, 1), th.rand(14, 1)
        self.assertAlmostEqual(float(r.loss("gci2", pos, neg)),
                               float(elbe.compute_loss(pos, neg)), places=6)

    def test_pairwise(self):
        r = _module_mode(PairwiseMarginRankingLoss(margin=1.0))
        # pairs: (0.5,2) (0.5,0) (1,3) (1,4) -> relu(1+p-n): 0, 1.5, 0, 0
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)), 1.5 / 4)

    def test_logsigmoid_uniform_and_adversarial(self):
        r = _module_mode(LogSigmoidLoss(margin=1.0))
        pos_term = -(F.logsigmoid(th.tensor(0.5)) + F.logsigmoid(th.tensor(0.0))) / 2
        neg0 = -(F.logsigmoid(th.tensor(1.0)) + F.logsigmoid(th.tensor(-1.0))) / 2
        neg1 = -(F.logsigmoid(th.tensor(2.0)) + F.logsigmoid(th.tensor(3.0))) / 2
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)),
                               float(pos_term + (neg0 + neg1) / 2), places=6)
        r = _module_mode(LogSigmoidLoss(margin=1.0, adversarial_temperature=1.0))
        w0 = th.softmax(-th.tensor([2.0, 0.0]), 0)
        w1 = th.softmax(-th.tensor([3.0, 4.0]), 0)
        neg0 = -(w0[0] * F.logsigmoid(th.tensor(1.0)) + w0[1] * F.logsigmoid(th.tensor(-1.0)))
        neg1 = -(w1[0] * F.logsigmoid(th.tensor(2.0)) + w1[1] * F.logsigmoid(th.tensor(3.0)))
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)),
                               float(pos_term + (neg0 + neg1) / 2), places=6)

    def test_bpr(self):
        r = _module_mode(BPRLoss())
        diffs = th.tensor([1.5, -0.5, 2.0, 3.0])
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)),
                               float(-F.logsigmoid(diffs).mean()), places=6)

    def test_logsumexp(self):
        r = _module_mode(LogSumExpContrastiveLoss(gamma1=0.75, gamma2=1.0))
        expected = (0.75 + 1.0) / 2 + math.log(3 * math.exp(0.0) + math.exp(1.0))
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)), expected, places=6)

    def test_pearson(self):
        r = _module_mode(PearsonCorrelationLoss(eps=0.0, include_positive_loss=False))
        s = th.exp(-th.tensor([0.5, 1.0, 2.0, 3.0, 0.0, 4.0]))
        t = th.tensor([1.0, 1, 0, 0, 0, 0])
        sc, tc = s - s.mean(), t - t.mean()
        rho = (sc * tc).sum() / (sc.norm() * tc.norm())
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)), float(1 - rho), places=6)
        # perfectly separated scores give a lower loss than reversed ones
        good = r.loss("gci0", th.tensor([0.0, 0.0]), th.tensor([5.0, 5.0]))
        bad = r.loss("gci0", th.tensor([5.0, 5.0]), th.tensor([0.0, 0.0]))
        self.assertLess(float(good), float(bad))

    def test_bce(self):
        r = _module_mode(ProbabilisticBCELoss())
        neg = th.tensor([2.0, 3.0, 0.5, 4.0])
        expected = 0.75 - th.log(1 - th.exp(-neg)).mean()
        self.assertAlmostEqual(float(r.loss("gci0", POS, neg.reshape(-1, 1))),
                               float(expected), places=5)


class TestRescoring(TestCase):

    def test_negatives_rescored_in_positive_form(self):
        m = StubModel({"gci0": [[0, 1], [2, 3]]})
        gen = FixedColumnCorruption(columns={"gci0": 1}, seed=0)
        loss = BPRLoss()
        regime = ComposedRegime(gen, loss)
        regime.attach(m)
        pos_ds = m.training_datasets["gci0"]
        neg = regime.filter_negatives("gci0", pos_ds, regime.negatives("gci0", pos_ds))
        pos_scores = m.module(pos_ds[:], "gci0")
        neg_scores_neg_form = m.module(neg, "gci0", neg=True)
        out = regime.loss("gci0", pos_scores, neg_scores_neg_form)
        diff = m.module(neg, "gci0").flatten() - pos_scores.flatten()
        expected = -F.logsigmoid(diff).mean()
        self.assertAlmostEqual(float(out), float(expected), places=6)

    def test_missing_negatives_raise(self):
        r = BPRLoss()
        r.attach(StubModel({"gci0": [[0, 1]]}))
        with self.assertRaises(RuntimeError):
            r.loss("gci0", POS, NEG)


class TestWeighting(TestCase):

    def test_frequency_weights(self):
        m = StubModel({"gci0": [[0, 1]] * 10, "gci2": [[0, 0, 1]] * 3}, batch_size=4)
        r = NormalFormFrequencyWeighting()
        r.attach(m)
        r.training_datasets(m.training_datasets)
        # ceil(10/4)=3, ceil(3/4)=1
        self.assertAlmostEqual(r.weights["gci0"], 0.75)
        self.assertAlmostEqual(r.weights["gci2"], 0.25)
        self.assertAlmostEqual(float(r.loss("gci2", POS, NEG)),
                               0.25 * float(m.compute_loss(POS, NEG)))
        r = NormalFormFrequencyWeighting(base=SumLoss(neg_weight=0.0), batch_size=1)
        r.attach(m)
        r.training_datasets(m.training_datasets)
        self.assertAlmostEqual(r.weights["gci0"], 10 / 13)
        self.assertAlmostEqual(float(r.loss("gci0", POS, NEG)), 10 / 13 * 0.75, places=6)


class TestEndToEnd(TestCase):

    def test_train_elbe_with_pearson_and_logsigmoid(self):
        from mowl.models import ELBE
        from tests.datasetFactory import FamilyDataset
        for loss in (PearsonCorrelationLoss(), LogSigmoidLoss(adversarial_temperature=1.0)):
            th.manual_seed(0)
            model = ELBE(FamilyDataset(), embed_dim=4)
            model.training_regime = ComposedRegime(
                FixedColumnCorruption(columns={"gci0": 1, "gci2": 2}, num_negatives=2,
                                      require_negative_loss=False, seed=0),
                loss)
            model.train(epochs=3, validate_every=100)
            for p in model.module.parameters():
                self.assertTrue(th.isfinite(p).all())
